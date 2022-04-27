import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class PrototypeAlignmentLearner(nn.Module):
    def __init__(self, way, shot, lazy_merge=False, ignore_label=255, temperature=20):
        super(PrototypeAlignmentLearner, self).__init__()
        self.way = way
        self.shot = shot
        self.ignore_label = ignore_label
        self.temperature = temperature
        self.eps = 1e-6
        self.lazy_merge = lazy_merge

    def forward(self, qry_feat, spt_feat, spt_mask, spt_ignore_idx):
        spt_mask = spt_mask.unsqueeze(1)
        spt_mask = F.interpolate(spt_mask.float(), spt_feat.size()[-2:], mode='bilinear', align_corners=True)

        qry_feat = rearrange(qry_feat, 'b c h w -> b 1 c h w')
        spt_feat = rearrange(spt_feat, '(b n s) c h w -> b n s c h w', n=self.way, s=self.shot)
        spt_mask = rearrange(spt_mask, '(b n s) 1 h w -> b n s 1 h w', n=self.way, s=self.shot)

        # support classes are assumed to be exclusive to each other
        if spt_ignore_idx is not None:
            spt_ignore_idx = spt_ignore_idx.unsqueeze(1)
            spt_ignore_idx = F.interpolate(spt_ignore_idx.float(), spt_feat.size()[-2:], mode='bilinear', align_corners=True)
            spt_ignore_idx = rearrange(spt_ignore_idx, '(b n s) 1 h w -> b n s 1 h w', n=self.way, s=self.shot)

            spt_mask_fg_count = torch.logical_and(spt_mask > 0, spt_ignore_idx != self.ignore_label).sum(dim=[2, -1, -2], keepdim=True).float()
            spt_mask_bg_count = torch.logical_and(spt_mask == 0, spt_ignore_idx != self.ignore_label).sum(dim=[2, -1, -2], keepdim=True).float()
            spt_mask_fg_binary = torch.logical_and(spt_mask > 0, spt_ignore_idx != self.ignore_label).float()
            spt_mask_bg_binary = torch.logical_and(spt_mask == 0, spt_ignore_idx != self.ignore_label).float()
        else:
            spt_mask_fg_count  = (spt_mask >  0).sum(dim=[2, -1, -2], keepdim=True).float()
            spt_mask_bg_count  = (spt_mask == 0).sum(dim=[2, -1, -2], keepdim=True).float()
            spt_mask_fg_binary = (spt_mask >  0).float()
            spt_mask_bg_binary = (spt_mask == 0).float()

        # B, N, S, C, H, W -> B, N, 1, C, 1, 1
        proto_fg = torch.sum(spt_feat * spt_mask_fg_binary, dim=[2, -1, -2], keepdim=True) / (spt_mask_fg_count + self.eps)
        # B, N, 1, C, 1, 1 -> B, N, C, 1, 1
        proto_fg = proto_fg.squeeze(2)

        if self.lazy_merge:
            '''
            This option enables the PANet to be trained/evaluated under the iFSL framework.
            But we leave it as an option since the authors originally propose to use the eager merge
            '''

            # B, N, S, C, H, W -> B, N, 1, C, 1, 1
            proto_bg = torch.sum(spt_feat * spt_mask_bg_binary, dim=[2, -1, -2], keepdim=True) / (spt_mask_bg_count + self.eps)
            # B, N, 1, C, 1, 1 -> B, N, C, 1, 1
            proto_bg = proto_bg.squeeze(2)

            ''' The episodic mask scheme proposed for iFSL '''
            # (B, 1, C, H, W), (B, N, C, 1, 1) -> B, N, H, W
            logit_mask_fg = F.cosine_similarity(qry_feat, proto_fg, dim=2) * self.temperature
            logit_mask_bg = F.cosine_similarity(qry_feat, proto_bg, dim=2) * self.temperature

            # B, 1 + N, C
            logit_mask = torch.cat((logit_mask_bg.mean(dim=1, keepdim=True), logit_mask_fg), dim=1)

        else: # eager merge
            # B, N, S, C, H, W -> B, 1, 1, C, 1, 1
            proto_bg = torch.sum(spt_feat * spt_mask_bg_binary, dim=[1, 2, -1, -2], keepdim=True) / (spt_mask_bg_count.sum(dim=1, keepdim=True) + self.eps)
            # B, 1, 1, C, 1, 1 -> B, 1, C, 1, 1
            proto_bg = proto_bg.squeeze(2)

            # B, 1 + N, C
            proto = torch.cat((proto_bg, proto_fg), dim=1)

            # (B, 1, C, H, W), (B, (1 + N), C, 1, 1) -> B, (1 + N), H, W
            logit_mask = F.cosine_similarity(qry_feat, proto, dim=2) * self.temperature

        return logit_mask
