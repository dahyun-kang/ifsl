from functools import reduce
from operator import add

import math
import torch

from torchvision.models import resnet

from einops import rearrange

from model.base.feature import extract_feat_res
from model.ifsl import iFSLModule
from model.module.panet import PrototypeAlignmentLearner


class PrototypeAlignmentNetwork(iFSLModule):
    def __init__(self, args):
        super(PrototypeAlignmentNetwork, self).__init__(args)

        # 1. Backbone network initialization
        self.backbone_type = args.backbone

        if args.backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = [16]
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif args.backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list([33])
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % args.backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.learner = PrototypeAlignmentLearner(args.way, args.shot)
        self.way = args.way

    def forward(self, batch):
        '''
        query_img.shape : [bsz, 3, H, W]
        support_imgs.shape : [bsz, way, shot, 3, H, W]
        support_masks.shape : [bsz, way, shot, H, W]
        '''
        support_imgs = rearrange(batch['support_imgs'], 'b n s c h w -> (b n s) c h w')
        support_masks = rearrange(batch['support_masks'], 'b n s h w -> (b n s) h w')
        support_ignore_idxs = batch.get('support_ignore_idxs')
        if support_ignore_idxs is not None:
            support_ignore_idxs = rearrange(batch['support_ignore_idxs'], 'b n s h w -> (b n s) h w')
        query_img = batch['query_img']

        query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats = self.extract_feats(support_imgs, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

        shared_masks = self.learner(query_feats[0], support_feats[0], support_masks, support_ignore_idxs)
        shared_masks = torch.log_softmax(shared_masks, dim=1)

        return shared_masks

    def predict_cls_and_mask(self, shared_masks, batch):
        ''' PANet with eager merging skips episodic bg merging step of Kang et al. 2022 '''
        # logit_seg = self.merge_bg_masks(shared_masks)
        logit_seg = shared_masks

        ''' The rest are the same with PFENet, HSNet and ASNet '''
        logit_seg = self.upsample_logit_mask(logit_seg, batch)

        with torch.no_grad():
            pred_cls = self.collect_class_presence(shared_masks)
            pred_seg = logit_seg.argmax(dim=1)

        return pred_cls, pred_seg, logit_seg

    def collect_class_presence(self, logit_mask):
        ''' logit_mask: B, (N + 1), H, W in the case of lazy_merge=False (default) '''
        # TODO: the categorical softmax above makes generalized class prediction extremely challenging..
        # do something
        assert self.way == 1
        class_activation = logit_mask[:, 1:].max(dim=-1)[0].max(dim=-1)[0] >= math.log(0.5)
        return class_activation.type(logit_mask.dtype).detach()

    def predict_mask_nshot(self, batch, nshot):
        shared_masks = self.forward(batch)
        pred_cls, pred_seg, _ = self.predict_cls_and_mask(shared_masks, batch)
        return pred_cls, pred_seg

    def train_mode(self):
        self.train()
        # self.backbone.eval()

    def configure_optimizers(self):
        ''' Taken from authors' official implementation '''
        return torch.optim.SGD(params=self.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=0.0005)
