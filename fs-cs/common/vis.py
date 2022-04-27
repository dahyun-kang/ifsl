r""" Visualize model predictions """
import os

from PIL import Image
import math
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

params= {'font.size': 18,
        'figure.figsize': (20, 8)  # W, H
}
plt.rcParams.update(params)

from . import utils


class Visualizer:
    """
    Visualizer class for segmentation prediction and GT
    """

    @classmethod
    def initialize(cls, visualize, way, path='./vis_results/'):
        cls.visualize = visualize
        if not visualize:
            return

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()
        cls.vis_path = path
        if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)
        cls.colors = [[23, 107, 239], # #4285f4
                      [255, 62, 48],  # #ea4335
                      [247, 181, 41], # #fbbc05
                      [23, 156, 82]]  # #34a853
        cls.colors = np.asarray(cls.colors)

        cls.way = way

    @classmethod
    def visualize_prediction_batch(cls, spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, qry_org_size_b, pred_mask_b, batch_idx, iou_b=None, er_b=None, to_cpu=False):
        if to_cpu:
            spt_img_b = utils.to_cpu(spt_img_b)
            spt_mask_b = utils.to_cpu(spt_mask_b)
            qry_img_b = utils.to_cpu(qry_img_b)
            qry_mask_b = utils.to_cpu(qry_mask_b)
            qry_org_size_b = utils.to_cpu(torch.hstack(qry_org_size_b).unsqueeze(0))
            pred_mask_b = utils.to_cpu(pred_mask_b)
        iou_b = [-1] * len(spt_img_b) if iou_b is None else iou_b
        er_b = [-1] * len(spt_img_b) if er_b is None else er_b

        for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, qry_org_size, pred_mask, iou, er) in \
                enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, qry_org_size_b, pred_mask_b, iou_b, er_b)):
            cls.visualize_prediction(spt_img, spt_mask, qry_img, qry_mask, qry_org_size, pred_mask, batch_idx, sample_idx, iou, er)

    @classmethod
    def to_numpy(cls, tensor, type):
        if type == 'img':
            return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception(f'Undefined tensor type: {type}')

    @classmethod
    def visualize_prediction(cls, spt_imgs, spt_masks, qry_img, qry_mask, qry_org_size, pred_mask, batch_idx, sample_idx, iou=None, er=None):

        spt_imgs = [cls.to_numpy(spt_img_c, 'img') for spt_img_c in spt_imgs]
        spt_masks = [cls.to_numpy(spt_mask_c, 'mask') for spt_mask_c in spt_masks]
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img_c, spt_mask_c, cls.colors)) for spt_img_c, spt_mask_c in zip(spt_imgs, spt_masks)]

        qry_img = cls.resize(qry_img, qry_org_size)
        qry_img = cls.to_numpy(qry_img, 'img')
        qry_mask = cls.to_numpy(qry_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask')
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), cls.colors))
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), cls.colors))

        vis_path = os.path.join(cls.vis_path, f'{batch_idx}_{sample_idx}.jpg')
        cls.save_plt(spt_masked_pils, pred_masked_pil, qry_masked_pil, iou, er, vis_path)

    @classmethod
    def apply_mask(cls, image, mask, color, alpha=0.5):
        r""" Apply mask to the given image. """

        for c in range(1, cls.way + 1):
            image[mask == c] = (1 - alpha) * image[mask == c] + alpha * color[c - 1]

        return image

    @classmethod
    def unnormalize(cls, img):
        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img

    @classmethod
    def resize(cls, img, spatial_size):
        img = img.clone()
        spatial_size = tuple([spatial_size[1].item(), spatial_size[0].item()])
        img = F.interpolate(img.unsqueeze(0), spatial_size, mode='bilinear', align_corners=True)
        return img.squeeze(0)

    @classmethod
    def save_plt(cls, spt_masked_pils, pred_masked_pil, qry_masked_pil, iou, er, vis_path):
        num_axes = len(spt_masked_pils) + 2
        plt.box(False)
        fig, axes = plt.subplots(1, num_axes)

        for i, spt in enumerate(spt_masked_pils):
            axes[i].set_title(f'support {i}')
            axes[i].imshow(spt)

        iou_str = 'x' if math.isnan(iou) else f'{iou:.1f}'
        axes[i + 1].set_title(f'pred\niou:{iou_str}\ner:{er:.1f}')
        axes[i + 1].imshow(pred_masked_pil)
        axes[i + 2].set_title(f'query GT')
        axes[i + 2].imshow(qry_masked_pil)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[], frame_on=False)
        plt.savefig(vis_path, bbox_inches='tight')
        plt.cla() ; plt.clf() ; plt.close()
