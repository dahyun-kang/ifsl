import os

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from common.evaluation import AverageMeter
from common import utils

from pprint import PrettyPrinter


class CustomTQDMProgressBar(TQDMProgressBar):
    def __init__(self):
        super(CustomTQDMProgressBar, self).__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        print('')
        for split in ['trn', 'val']:
            loss = trainer.callback_metrics[f'{split}/loss']
            miou = trainer.callback_metrics[f'{split}/miou']
            er   = trainer.callback_metrics[f'{split}/er']

            print(f'[{split}] ep: {trainer.current_epoch:>3}| {split}/loss: {loss:.3f} | {split}/miou: {miou:.3f} | {split}/er: {er:.3f}')

    def get_progress_bar_dict(self):
        # to stop to show the version number in the progress bar
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


class MeterCallback(Callback):
    # TODO: this class should cotain non-side-effectivce functions like printing
    # otherwise, move them to LightningModule
    """
    A class that initiates classificaiton and segmentation metrics
    """
    def __init__(self, args):
        super(MeterCallback, self).__init__()
        self.args = args

    def on_fit_start(self, trainer, pl_module):
        PrettyPrinter().pprint(vars(self.args))
        print(pl_module.learner)
        utils.print_param_count(pl_module)

        if not self.args.nowandb and not self.args.eval:
            trainer.logger.experiment.watch(pl_module)

    def on_test_start(self, trainer, pl_module):
        PrettyPrinter().pprint(vars(self.args))
        utils.print_param_count(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        utils.fix_randseed(None)
        dataset = trainer.train_dataloader.dataset.datasets
        pl_module.trn_average_meter = AverageMeter(dataset, self.args.way)
        pl_module.train_mode()

    def on_validation_epoch_start(self, trainer, pl_module):
        self._shared_eval_epoch_start(trainer.val_dataloaders[0].dataset, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        self._shared_eval_epoch_start(trainer.test_dataloaders[0].dataset, pl_module)

    def _shared_eval_epoch_start(self, dataset, pl_module):
        utils.fix_randseed(0)
        pl_module.val_average_meter = AverageMeter(dataset, self.args.way)
        pl_module.eval()


class CustomCheckpoint(ModelCheckpoint):
    """
    Checkpoint load & save
    """
    def __init__(self, args):
        self.dirpath = os.path.join('logs', args.benchmark, f'fold{args.fold}', args.backbone, args.logpath)
        if not args.eval and not args.resume:
            assert not os.path.exists(self.dirpath), f'{self.dirpath} already exists'
        self.filename = 'best_model'
        self.way = args.way
        self.weak = args.weak
        self.monitor = 'val/er' if args.weak else 'val/miou'

        super(CustomCheckpoint, self).__init__(dirpath=self.dirpath,
                                               monitor=self.monitor,
                                               filename=self.filename,
                                               mode='max',
                                               verbose=True,
                                               save_last=True)
        # For evaluation, load best_model-v(k).cpkt where k is the max index
        if args.eval:
            self.modelpath = self.return_best_model_path(self.dirpath, self.filename)
            print('evaluating', self.modelpath)
        # For training, set the filename as best_model.ckpt
        # For resuming training, pytorch_lightning will automatically set the filename as best_model-v(k).ckpt
        else:
            self.modelpath = os.path.join(self.dirpath, self.filename + '.ckpt')
        self.lastmodelpath = os.path.join(self.dirpath, 'last.ckpt') if args.resume else None

    def return_best_model_path(self, dirpath, filename):
        ckpt_files = os.listdir(dirpath)  # list of strings
        vers = [ckpt_file for ckpt_file in ckpt_files if filename in ckpt_file]
        vers.sort()
        # vers = ['best_model.ckpt'] or
        # vers = ['best_model-v1.ckpt', 'best_model-v2.ckpt', 'best_model.ckpt']
        best_model = vers[-1] if len(vers) == 1 else vers[-2]
        return os.path.join(self.dirpath, best_model)


class OnlineLogger(WandbLogger):
    """
    A wandb logger class that is customed with the experiment log path
    """
    def __init__(self, args):
        super(OnlineLogger, self).__init__(
            name=args.logpath,
            project=f'fscs-{args.benchmark}-{args.backbone}',
            group=f'fold{args.fold}',
            log_model=False,
        )
        self.experiment.config.update(args)
