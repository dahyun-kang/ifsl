import torch
import torch.nn as nn
import torch.nn.functional as F


class PFENetLearner(nn.Module):
    def __init__(self, way, shot, ppm_scales=[50, 25, 13, 7]):
        super(PFENetLearner, self).__init__()
        self.way = way
        self.shot = shot
        self.eps = 1e-6
        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        reduce_dim = 256
        fea_dim = 1024 + 512
        factor = 1
        mask_add_num = 1
        classes = 2
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

    def forward(self, qry_feat, spt_feat, corr_query_mask):
        out_list = []
        pyramid_feat_list = []

        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(qry_feat.shape[2] * tmp_bin)
                qry_feat_bin = nn.AdaptiveAvgPool2d(bin)(qry_feat)
            else:
                bin = tmp_bin
                qry_feat_bin = self.avgpool_list[idx](qry_feat)
            spt_feat_bin = spt_feat.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            if self.way > 1:
                qry_feat_bin = qry_feat_bin.repeat_interleave(self.way, dim=0)
                corr_mask_bin = corr_mask_bin.repeat_interleave(self.way, dim=0)
            merge_feat_bin = torch.cat([qry_feat_bin, spt_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(qry_feat.size(2), qry_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        qry_feat = torch.cat(pyramid_feat_list, 1)
        qry_feat = self.res1(qry_feat)
        qry_feat = self.res2(qry_feat) + qry_feat
        logit_mask = self.cls(qry_feat)

        logit_mask = logit_mask.view(-1, self.way, *logit_mask.shape[1:])

        # B, N, 2, H, W
        return logit_mask
