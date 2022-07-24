import math
import torch
import pytorch_lightning as pl
import torch.nn.functional as F


class iFSLModule(pl.LightningModule):
    """
    """
    def __init__(self, args):
        super(iFSLModule, self).__init__()

        self.args = args
        self.way = self.args.way
        self.weak = args.weak
        self.range = torch.arange(args.way + 1, requires_grad=False).view(1, args.way + 1, 1, 1)
        self.learner = None

    def forward(self, batch):
        pass

    def train_mode(self):
        pass

    def configure_optimizers(self):
        pass

    def predict_mask_nshot(self, batch, nshot):
        pass

    def training_step(self, batch, batch_idx):
        self.average_meter = self.trn_average_meter
        return self.shared_step(batch, batch_idx, 'trn')

    def shared_step(self, batch, batch_idx, split):
        """
        batch.keys()
        > dict_keys(['query_img', 'query_mask', 'query_name', 'query_ignore_idx', 'org_query_imsize', 'support_imgs', 'support_masks', 'support_names', 'support_ignore_idxs', 'class_id'])

        batch['query_img'].shape : [bsz, 3, H, W]
        batch['query_mask'].shape : [bsz, H, W]
        batch['query_name'].len : [bsz]
        batch['query_ignore_idx'].shape : [bsz, H, W]
        batch['query_ignore_idx'].shape : [bsz, H, W]
        batch['org_query_imsize'].len : [bsz]
        batch['support_imgs'].shape : [bsz, way, shot, 3, H, W]
        batch['support_masks'].shape : [bsz, way, shot, H, W]
        # FYI: this support_names' shape is transposed so keep in mind for vis
        batch['support_names'].shape : [bsz, shot, way]
        batch['support_ignore_idxs'].shape: [bsz, way, shot, H, W]
        batch['class_id'].shape : [bsz]
        batch['support_classes'].shape : [bsz, way] (torch.int64)
        batch['query_class_presence'].shape : [bsz, way] (torch.bool)
        # FYI: K-shot is always fixed to 1 for training
        """

        shared_masks = self.forward(batch)
        pred_cls, pred_seg, logit_seg = self.predict_cls_and_mask(shared_masks, batch)

        if self.weak:
            loss = self.compute_cls_objective(shared_masks, batch['query_class_presence'])
        else:
            loss = self.compute_seg_objective(logit_seg, batch['query_mask'])

        with torch.no_grad():
            self.average_meter.update_cls(pred_cls, batch['query_class_presence'])
            self.average_meter.update_seg(pred_seg, batch, loss.item())

            self.log(f'{split}/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self._shared_epoch_end(training_step_outputs, 'trn')

    def validation_step(self, batch, batch_idx):
        # model.eval() and torch.no_grad() are called automatically for validation
        # in pytorch_lightning
        self.average_meter = self.val_average_meter
        self.shared_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, validation_step_outputs):
        # model.eval() and torch.no_grad() are called automatically for validation
        # in pytorch_lightning
        self._shared_epoch_end(validation_step_outputs, 'val')

    def _shared_epoch_end(self, steps_outputs, split):
        self.average_meter = self.trn_average_meter if split == 'trn' else self.val_average_meter

        miou = self.average_meter.compute_iou()
        er = self.average_meter.compute_cls_er()
        loss = self.average_meter.avg_seg_loss()

        dict = {f'{split}/loss': loss,
                f'{split}/miou': miou,
                f'{split}/er': er}

        for k in dict:
            self.log(k, dict[k], on_epoch=True, logger=True)

        # Moved to common/callback.py due to the pytorch-lightning update
        # space = '\n\n' if split == 'val' else '\n'
        # self.print(f'{space}[{split}] ep: {self.current_epoch:>3}| {split}/loss: {loss:.3f} | {split}/miou: {miou:.3f} | {split}/er: {er:.3f}')

    def test_step(self, batch, batch_idx):
        pred_cls, pred_seg = self.predict_mask_nshot(batch, self.args.shot)
        er_b = self.average_meter.update_cls(pred_cls, batch['query_class_presence'], loss=None)
        iou_b = self.average_meter.update_seg(pred_seg, batch, loss=None)

        if self.args.vis:
            print(batch_idx, 'qry:', batch['query_name'])
            print(batch_idx, 'spt:', batch['support_names'])
            if self.args.shot > 1: raise NotImplementedError
            if self.args.weak:
                batch['support_masks'] = torch.zeros(1, self.way, 400, 400).cuda()
            from common.vis import Visualizer
            Visualizer.initialize(True, self.way)
            Visualizer.visualize_prediction_batch(batch['support_imgs'].squeeze(2),
                                                  batch['support_masks'].squeeze(2),
                                                  batch['query_img'],
                                                  batch['query_mask'],
                                                  batch['org_query_imsize'],
                                                  pred_seg,
                                                  batch_idx,
                                                  iou_b=iou_b,
                                                  er_b=er_b,
                                                  to_cpu=True)

    def test_epoch_end(self, test_step_outputs):
        miou = self.average_meter.compute_iou()
        er = self.average_meter.compute_cls_er()
        length = 16
        dict = {'benchmark'.ljust(length): self.args.benchmark,
                'fold'.ljust(length): self.args.fold,
                'test/miou'.ljust(length): miou.item(),
                'test/er'.ljust(length): er.item()}

        for k in dict:
            self.log(k, dict[k], on_epoch=True)

    def predict_cls_and_mask(self, shared_masks, batch):
        logit_seg = self.merge_bg_masks(shared_masks)
        logit_seg = self.upsample_logit_mask(logit_seg, batch)

        with torch.no_grad():
            pred_cls = self.collect_class_presence(shared_masks)
            pred_seg = logit_seg.argmax(dim=1)

        return pred_cls, pred_seg, logit_seg

    def collect_class_presence(self, logit_mask):
        ''' logit_mask: B, N, 2, H, W '''
        # since logit_mask is log-softmax-ed, we use torch.log(0.5) for the threshold
        class_activation = logit_mask[:, :, 1].max(dim=-1)[0].max(dim=-1)[0] >= math.log(0.5)
        return class_activation.type(logit_mask.dtype).detach()

    def upsample_logit_mask(self, logit_mask, batch):
        if self.training:
            spatial_size = batch['query_img'].shape[-2:]
        else:
            spatial_size = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
        return F.interpolate(logit_mask, spatial_size, mode='bilinear', align_corners=True)

    def compute_seg_objective(self, logit_mask, gt_mask):
        ''' supports 1-way training '''
        return F.nll_loss(logit_mask, gt_mask.long())

    def compute_cls_objective(self, shared_masks, gt_presence):
        ''' supports 1-way training '''
        # B, N, 2, H, W -> B, N, 2 -> B, 2
        prob_avg = shared_masks.mean(dim=[-1, -2]).squeeze(1)
        return F.nll_loss(prob_avg, gt_presence.long().squeeze(-1))

    def merge_bg_masks(self, shared_fg_masks):
        # B, N, H, W
        logit_fg = shared_fg_masks[:, :, 1]
        # B, 1, H, W
        logit_episodic_bg = shared_fg_masks[:, :, 0].mean(dim=1)
        # B, (1 + N), H, W
        logit_mask = torch.cat((logit_episodic_bg.unsqueeze(1), logit_fg), dim=1)

        return logit_mask


