r""" Dataloader builder for few-shot classification and segmentation task """
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO


class FSCSDatasetModule(LightningDataModule):
    """
    A LightningDataModule for FS-CS benchmark
    """
    def __init__(self, args, img_size=400):
        super().__init__()
        self.args = args
        self.datapath = args.datapath

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        self.img_size = img_size
        self.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
        }
        self.transform = transforms.Compose([transforms.Resize(size=(self.img_size, self.img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.img_mean, self.img_std)])

    def train_dataloader(self):
        dataset = self.datasets[self.args.benchmark](self.datapath,
                                                     fold=self.args.fold,
                                                     transform=self.transform,
                                                     split='trn',
                                                     way=self.args.way,
                                                     shot=1)  # shot=1 fixed for training
        dataloader = DataLoader(dataset, batch_size=self.args.bsz, shuffle=True, num_workers=8)
        return dataloader

    def val_dataloader(self):
        dataset = self.datasets[self.args.benchmark](self.datapath,
                                                     fold=self.args.fold,
                                                     transform=self.transform,
                                                     split='val',
                                                     way=self.args.way,
                                                     shot=self.args.shot)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
