
import torch
import argparse

from pytorch_lightning import Trainer

from model.asnet import AttentiveSqueezeNetwork
from data.dataset import FSSDatasetModule
from common.callbacks import MeterCallback, CustomProgressBar, CustomCheckpoint, OnlineLogger


def main(args):

    # Dataset initialization
    dm = FSSDatasetModule(args)

    # Pytorch-lightning main trainer
    checkpoint_callback = CustomCheckpoint(args)
    trainer = Trainer(accelerator='dp',  # DataParallel
                      callbacks=[MeterCallback(args), CustomCheckpoint(args), CustomProgressBar()],
                      gpus=torch.cuda.device_count(),
                      logger=False if args.nowandb or args.eval else OnlineLogger(args),
                      progress_bar_refresh_rate=1,
                      max_epochs=args.niter,
                      num_sanity_val_steps=0,
                      weights_summary=None,
                      resume_from_checkpoint=checkpoint_callback.lastmodelpath,
                      # profiler='advanced',  # this awesome profiler is easy to use
                      )

    if args.eval:
        # Loading the best model checkpoint from args.logpath
        modelpath = checkpoint_callback.modelpath
        model = AttentiveSqueezeNetwork.load_from_checkpoint(modelpath, args=args, use_original_imgsize=args.use_original_imgsize)
        trainer.test(model, test_dataloaders=dm.test_dataloader())
    else:
        # Train
        model = AttentiveSqueezeNetwork(args, False)
        trainer.fit(model, dm)


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Pytorch Implementation of Attentive Squeeze Network for FSS')
    parser.add_argument('--datapath', type=str, default='~/datasets', help='Dataset path containing the root dir of pascal & coco')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco'], help='Experiment benchmark')
    parser.add_argument('--logpath', type=str, default='', help='Checkpoint saving dir identifier')
    parser.add_argument('--shot', type=int, default=1, help='K-shot for K-shot evaluation episode: fixed to 1 for training')
    parser.add_argument('--bsz', type=int, default=12, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--niter', type=int, default=2000, help='Max iterations')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3], help='4-fold validation fold')
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101'], help='Backbone CNN network')
    parser.add_argument('--nowandb', action='store_true', help='Flag not to log at wandb')
    parser.add_argument('--use_original_imgsize', action='store_true', help='Flag to evaluate the seg. result in the original image size')
    parser.add_argument('--eval', action='store_true', help='Flag to evaluate a model checkpoint')
    parser.add_argument('--resume', action='store_true', help='Flag to resume a finished run')
    args = parser.parse_args()

    main(args)
