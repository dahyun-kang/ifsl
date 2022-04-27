r""" Pascal-5i few-shot classification and segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetPASCAL(Dataset):
    """
    FS-CS Pascal-5i dataset of which split follows the standard FS-S dataset
    """
    def __init__(self, datapath, fold, transform, split, way, shot):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor')

        self.PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
                [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
                [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

        self.fold = fold
        self.way = way
        self.shot = shot

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        query_name, support_names, _support_classes = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_class_presence = [s_c in torch.unique(query_cmask) for s_c in _support_classes]  # needed - 1
        rename_class = lambda x: _support_classes.index(x) + 1 if x in _support_classes else 0

        query_img = self.transform(query_img)
        query_mask, query_ignore_idx = self.get_query_mask(query_img, query_cmask, rename_class)
        support_imgs = torch.stack([torch.stack([self.transform(support_img) for support_img in support_imgs_c]) for support_imgs_c in support_imgs])
        support_masks, support_ignore_idxs = self.get_support_masks(support_imgs, _support_classes, support_cmasks, rename_class)

        _support_classes = torch.tensor(_support_classes)
        query_class_presence = torch.tensor(query_class_presence)

        assert query_class_presence.int().sum() == (len(torch.unique(query_mask)) - 1)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'support_classes': _support_classes,
                 'query_class_presence': query_class_presence}

        return batch

    def get_query_mask(self, query_img, query_cmask, rename_class):
        if self.split == 'trn':  # resize during training and retain orignal sizes during validation
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.generate_query_episodic_mask(query_cmask.float(), rename_class)
        return query_mask, query_ignore_idx

    def get_support_masks(self, support_imgs, _support_classes, support_cmasks, rename_class):
        support_masks = []
        support_ignore_idxs = []
        for class_id, scmask_c in zip(_support_classes, support_cmasks):  # ways
            support_masks_c = []
            support_ignore_idxs_c = []
            for scmask in scmask_c:  # shots
                scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
                support_mask, support_ignore_idx = self.generate_support_episodic_mask(scmask, class_id, rename_class)
                assert len(torch.unique(support_mask)) <= 2, f'{len(torch.unique(support_mask))} labels in support'
                support_masks_c.append(support_mask)
                support_ignore_idxs_c.append(support_ignore_idx)
            support_masks.append(torch.stack(support_masks_c))
            support_ignore_idxs.append(torch.stack(support_ignore_idxs_c))
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)
        return support_masks, support_ignore_idxs

    def generate_query_episodic_mask(self, mask, rename_class):
        # mask = mask.clone()
        mask_renamed = torch.zeros_like(mask).to(mask.device).type(mask.dtype)
        boundary = (mask / 255).floor()

        classes = torch.unique(mask)
        for c in classes:
            mask_renamed[mask == c] = 0 if c in [0, 255] else rename_class(c)

        return mask_renamed, boundary

    def generate_support_episodic_mask(self, mask, class_id, rename_class):
        mask = mask.clone()
        boundary = (mask / 255).floor()
        mask[mask != class_id] = 0
        mask[mask == class_id] = rename_class(class_id)

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img  = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs  = [[self.read_img(name)  for name in support_names_c] for support_names_c in support_names]
        support_masks = [[self.read_mask(name) for name in support_names_c] for support_names_c in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        # Fix (q, s) pair for all queries across different batch sizes for reproducibility
        if self.split == 'val':
            np.random.seed(idx)

        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, query_class = self.img_metadata[idx]

        # 3-way 2-shot support_names: [[c1_1, c1_2], [c2_1, c2_2], [c3_1, c3_2]]
        support_names = []

        # p encourage the support classes sampled as the query_class by the prob of 0.5
        p = np.ones([len(self.class_ids)]) / 2. / float(len(self.class_ids) - 1)
        p[self.class_ids.index(query_class)] = 1 / 2.
        support_classes = np.random.choice(self.class_ids, self.way, p=p, replace=False).tolist()
        for sc in support_classes:
            support_names_c = []
            while True:  # keep sampling support set if query == support
                support_name = np.random.choice(self.img_metadata_classwise[sc], 1, replace=False)[0]
                if query_name != support_name and support_name not in support_names_c:
                    support_names_c.append(support_name)
                if len(support_names_c) == self.shot:
                    break
            support_names.append(support_names_c)

        return query_name, support_names, support_classes

    def build_class_ids(self):
        nclass_val = self.nclass // self.nfolds
        # e.g. fold0 val: 1, 2, 3, 4, 5
        class_ids_val = [self.fold * nclass_val + i for i in range(1, nclass_val + 1)]
        # e.g. fold0 trn: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
        class_ids_trn = [x for x in range(1, self.nclass + 1) if x not in class_ids_val]

        assert len(set(class_ids_trn + class_ids_val)) == self.nclass
        assert 0 not in class_ids_val
        assert 0 not in class_ids_trn

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join(f'data/splits/pascal/{split}/fold{fold_id}.txt')
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1])] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print(f'Total {self.split} images are : {len(img_metadata):,}')

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(1, self.nclass + 1):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]

        # img_metadata_classwise.keys(): [1, 2, ..., 20]
        assert 0 not in img_metadata_classwise.keys()
        assert self.nclass in img_metadata_classwise.keys()

        return img_metadata_classwise
