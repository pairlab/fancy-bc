import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import json
import torch.nn.functional as F
from robomimic.algo import algo_factory
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings

class RobomimicLightning(pl.LightningModule):
    def __init__(self, config):
        super(RobomimicLightning, self).__init__()
        self.config = config
        self.model = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=config.shape_meta_list[0]["all_shapes"],
            ac_dim=config.shape_meta_list[0]["ac_dim"],
            device=self.device,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Process batch for training
        input_batch = self.model.process_batch_for_training(batch)
        input_batch = self.model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)

        # Forward and backward pass
        info = self.model.train_on_batch(input_batch, self.current_epoch, validate=False)

        # Log the loss
        step_log = self.model.log_info(info)
        self.log('train_loss', step_log['loss'])
        return step_log['loss']

    def validation_step(self, batch, batch_idx):
        # Process batch for validation
        input_batch = self.model.process_batch_for_training(batch)
        input_batch = self.model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)

        # Forward pass
        info = self.model.train_on_batch(input_batch, self.current_epoch, validate=True)

        # Log the loss
        step_log = self.model.log_info(info)
        self.log('val_loss', step_log['loss'])
        return step_log['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.train.lr)
        return optimizer


def main(args):
    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        dataset_paths = args.dataset.split(',')
        config.train.data = [{"path": path} for path in dataset_paths]

    if args.name is not None:
        config.experiment.name = args.name

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    if args.debug:
        config.unlock()
        config.lock_keys()
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10
        config.train.output_dir = "/tmp/tmp_trained_models"

    config.lock()

    # Initialize datasets and dataloaders
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=config.shape_meta["all_obs_keys"])
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_data_workers,
        drop_last=True
    )
    if validset is not None:
        valid_loader = DataLoader(
            dataset=validset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_data_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    model = RobomimicLightning(config)
    trainer = pl.Trainer(
        max_epochs=config.train.num_epochs,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        strategy='ddp'
    )
    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="(optional) path to a config json that will be used to override the default settings.")
    parser.add_argument("--algo", type=str, help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided")
    parser.add_argument("--name", type=str, default=None, help="(optional) if provided, override the experiment name defined in the config")
    parser.add_argument("--dataset", type=str, default=None, help="(optional) if provided, override the dataset path defined in the config")
    parser.add_argument("--debug", action='store_true', default=False, help="(optional) if provided, override the dataset path defined in the config")
    args = parser.parse_args()
    main(args)

