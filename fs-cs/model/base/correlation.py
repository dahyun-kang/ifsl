r""" Provides functions that builds/manipulates correlation tensors """
import torch


class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids, way=1):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bszs, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bszs, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bszq, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bszq, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            if way > 1:
                query_feat = query_feat.repeat_interleave(way, dim=0)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bszs, ha, wa, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]
