import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from einops import rearrange

from model.ifsl import iFSLModule
from model.module.pfenet import PFENetLearner


class PriorGuidedFeatureEnrichmentNetwork(iFSLModule):
    def __init__(self, args):
        super(PriorGuidedFeatureEnrichmentNetwork, self).__init__(args)

        self.way = args.way
        self.shot = args.shot
        self.ignore_label = 255

        # 1. Backbone network initialization
        self.backbone_type = args.backbone
        if args.backbone in ['resnet50', 'resnet101']:
            backbone = resnet.resnet50(pretrained=True)
            self.backbone0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
            self.backbone1, self.backbone2, self.backbone3, self.backbone4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
            del backbone
        elif args.backbone == 'resnet101':
            raise NotImplementedError
        else:
            raise Exception('Unavailable backbone: %s' % args.backbone)

        for n, m in self.backbone3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.backbone4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        reduce_dim = 256
        fea_dim = 1024 + 512
        self.down_qry = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_spt = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.backbone0.eval()
        self.backbone1.eval()
        self.backbone2.eval()
        self.backbone3.eval()
        self.backbone4.eval()
        self.learner = PFENetLearner(args.way, args.shot)

    def forward(self, batch):
        '''
        query_img.shape : [bsz, 3, H, W]
        support_imgs.shape : [bsz, way, shot, 3, H, W]
        support_masks.shape : [bsz, way, shot, H, W]
        '''

        with torch.no_grad():
            support_imgs = rearrange(batch['support_imgs'], 'b n s c h w -> (b n) s c h w')
            support_masks = rearrange(batch['support_masks'], 'b n s h w -> (b n) s h w')
            support_ignore_idxs = batch.get('support_ignore_idxs')
            if support_ignore_idxs is not None:
                support_ignore_idxs = rearrange(batch['support_ignore_idxs'], 'b n s h w -> (b n) s h w')
            query_img = batch['query_img']

        qry_feat, qry_feat4 = self.extract_qry_feat(query_img)
        spt_feat, spt_feat4, mask_list = self.extract_spt_feat(support_imgs, support_masks, support_ignore_idxs)
        corr_query_mask = self.correlation_query_mask(qry_feat, qry_feat4, spt_feat4, mask_list)

        shared_masks = self.learner(qry_feat, spt_feat, corr_query_mask)

        # B, N, 2, H, W
        shared_masks = torch.log_softmax(shared_masks, dim=2)
        return shared_masks

    def predict_mask_nshot(self, batch, nshot):
        shared_masks = self.forward(batch)
        pred_cls, pred_seg, _ = self.predict_cls_and_mask(shared_masks, batch)
        return pred_cls, pred_seg

    def train_mode(self):
        self.train()
        self.backbone0.eval()
        self.backbone1.eval()
        self.backbone2.eval()
        self.backbone3.eval()
        self.backbone4.eval()

    def configure_optimizers(self):
        # adopted from authors' official implementation
        return torch.optim.SGD(params=[
        {'params': self.down_qry.parameters()},
        {'params': self.down_spt.parameters()},
        {'params': self.learner.init_merge.parameters()},
        {'params': self.learner.alpha_conv.parameters()},
        {'params': self.learner.beta_conv.parameters()},
        {'params': self.learner.inner_cls.parameters()},
        {'params': self.learner.res1.parameters()},
        {'params': self.learner.res2.parameters()},
        {'params': self.learner.cls.parameters()}], lr=0.0025, momentum=0.9, weight_decay=0.0001)

    def extract_qry_feat(self, x):
        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.backbone0(x)
            query_feat_1 = self.backbone1(query_feat_0)
            query_feat_2 = self.backbone2(query_feat_1)
            query_feat_3 = self.backbone3(query_feat_2)
            query_feat_4 = self.backbone4(query_feat_3)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_qry(query_feat)
        return query_feat, query_feat_4

    def extract_spt_feat(self, s_x, s_y, s_ignore):
        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        for i in range(self.shot):
            if s_ignore is not None:
                mask = torch.logical_and(s_y[:, i, :, :] > 0, s_ignore[:, i, :, :] != self.ignore_label).float().unsqueeze(1)
            else:
                mask = s_y[:, i, :, :].float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.backbone0(s_x[:, i, :, :, :])
                supp_feat_1 = self.backbone1(supp_feat_0)
                supp_feat_2 = self.backbone2(supp_feat_1)
                supp_feat_3 = self.backbone3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.backbone4(supp_feat_3 * mask)
                final_supp_list.append(supp_feat_4)

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_spt(supp_feat)
            supp_feat = self.weighted_GAP(supp_feat, mask)
            supp_feat_list.append(supp_feat)

        if self.shot > 1:
            supp_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
            supp_feat /= len(supp_feat_list)

        return supp_feat, final_supp_list, mask_list

    def weighted_GAP(self, supp_feat, mask):
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat

    def correlation_query_mask(self, qry_feat, qry_feat4, spt_feat4, mask_list):
        corr_query_mask_list = []
        for i, tmp_supp_feat in enumerate(spt_feat4):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = qry_feat4
            s = tmp_supp_feat_4
            corr_query = self.cross_correlation(q, s, spatial_size=(qry_feat.size()[2], qry_feat.size()[3]))
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
        corr_query_mask = F.interpolate(corr_query_mask, size=(qry_feat.size(2), qry_feat.size(3)), mode='bilinear', align_corners=True)

        return corr_query_mask

    def cross_correlation(self, q, s, spatial_size):
        cosine_eps = 1e-7
        bsize, ch_sz, sp_sz, _ = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
        similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
        corr_query = F.interpolate(corr_query, size=spatial_size, mode='bilinear', align_corners=True)
        return corr_query
