
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.base.aslayer import AttentiveSqueezeLayer


class AttentionLearner(nn.Module):
    def __init__(self, inch):
        super(AttentionLearner, self).__init__()

        def make_building_attentive_block(in_channel, out_channels, kernel_sizes, spt_strides, pool_kv=False):
            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                padding = ksz // 2 if ksz > 2 else 0
                building_block_layers.append(AttentiveSqueezeLayer(inch, outch, ksz, stride, padding, pool_kv=pool_kv))

            return nn.Sequential(*building_block_layers)

        self.feat_ids = list(range(4, 17))

        self.encoder_layer4 = make_building_attentive_block(inch[0], [32, 128], [5, 3], [4, 2])
        self.encoder_layer3 = make_building_attentive_block(inch[1], [32, 128], [5, 5], [4, 4], pool_kv=True)
        self.encoder_layer2 = make_building_attentive_block(inch[2], [32, 128], [5, 5], [4, 4], pool_kv=True)

        self.encoder_layer4to3 = make_building_attentive_block(128, [128, 128], [1, 2], [1, 1])
        self.encoder_layer3to2 = make_building_attentive_block(128, [128, 128], [1, 2], [1, 1])

        # Decoder layers
        self.decoder1 = nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_query_dims(self, hypercorr, spatial_size):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = rearrange(hypercorr, 'b c d t h w -> (b h w) c d t')
        # (B H W) C D T -> (B H W) C * spatial_size
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        return rearrange(hypercorr, '(b h w) c d t -> b c d t h w', b=bsz, h=hb, w=wb)

    def forward(self, hypercorr_pyramid, support_mask):

        hypercorr_sqz4 = self.encoder_layer4((hypercorr_pyramid[0], support_mask))[0]
        hypercorr_sqz3 = self.encoder_layer3((hypercorr_pyramid[1], support_mask))[0]
        hypercorr_sqz2 = self.encoder_layer2((hypercorr_pyramid[2], support_mask))[0]

        hypercorr_sqz4 = hypercorr_sqz4.mean(dim=[-1, -2], keepdim=True)
        hypercorr_sqz4 = self.interpolate_query_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3((hypercorr_mix43, support_mask))[0]

        hypercorr_mix43 = self.interpolate_query_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2((hypercorr_mix432, support_mask))[0]

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).squeeze(-1)

        hypercorr_decoded = self.decoder1(hypercorr_encoded)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.decoder2(hypercorr_decoded)

        return logit_mask
