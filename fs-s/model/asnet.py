
from functools import reduce
from operator import add

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision.models import resnet
from torchvision.models import vgg

from model.base.feature import extract_feat_vgg, extract_feat_res
from model.base.correlation import Correlation
from common import utils
from common.evaluation import Evaluator

from model.learner import AttentionLearner


class AttentiveSqueezeNetwork(pl.LightningModule):
    def __init__(self, args, use_original_imgsize):
        super(AttentiveSqueezeNetwork, self).__init__()

        self.args = args

        # 1. Backbone network initialization
        self.backbone_type = args.backbone
        self.use_original_imgsize = use_original_imgsize

        if args.backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif args.backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif args.backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % args.backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.learner = AttentionLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = F.cross_entropy

    def forward(self, query_img, support_img, support_mask):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids, pool=True)
            corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)

        logit_mask = self.learner(corr, support_mask)
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit_mask

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging

    def training_step(self, batch, batch_idx):
        split = 'trn' if self.training else 'val'
        logit_mask = self.forward(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)

        loss = self.compute_objective(logit_mask, batch['query_mask'])

        with torch.no_grad():
            area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
            self.average_meter.update(area_inter.cpu(), area_union.cpu(), batch['class_id'].cpu(), loss.item())

            self.log(f'{split}/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self._shared_epoch_end(training_step_outputs)

    def validation_step(self, batch, batch_idx):
        # model.eval() and torch.no_grad() are called automatically for validation
        # in pytorch_lightning
        self.training_step(batch, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        # model.eval() and torch.no_grad() are called automatically for validation
        # in pytorch_lightning
        self._shared_epoch_end(validation_step_outputs)

    def _shared_epoch_end(self, steps_outputs):
        split = 'trn' if self.training else 'val'
        avg_loss = utils.mean(self.average_meter.loss_buf)
        miou, fb_iou = self.average_meter.compute_iou()

        dict = {f'{split}/loss': avg_loss,
                f'{split}/miou': miou,
                f'{split}/fb_iou': fb_iou}

        for k in dict:
            self.log(k, dict[k], on_epoch=True, logger=True)

        space = '\n\n' if split == 'val' else '\n'
        print(f'{space}[{split}] ep: {self.current_epoch:>3}| {split}/loss: {avg_loss:.3f} | {split}/miou: {miou:.3f} | {split}/fb_iou: {fb_iou:.3f}')

    def test_step(self, batch, batch_idx):
        pred_mask = self.predict_mask_nshot(batch, self.args.shot)
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        self.average_meter.update(area_inter.cpu(), area_union.cpu(), batch['class_id'].cpu(), loss=None)

    def test_epoch_end(self, test_step_outputs):
        miou, fb_iou = self.average_meter.compute_iou()
        length = 16
        dict = {'benchmark'.ljust(length): self.args.benchmark,
                'fold'.ljust(length): self.args.fold,
                'test/miou'.ljust(length): miou.item(),
                'test/fb_iou'.ljust(length): fb_iou.item()}

        for k in dict:
            self.log(k, dict[k], on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam([{"params": self.parameters(), "lr": self.args.lr}])

    def get_progress_bar_dict(self):
        # to stop to show the version number in the progress bar
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
