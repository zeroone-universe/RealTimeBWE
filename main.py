
from datamodule import RTBWEDataModule

from train import RTBWETrain

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import *
import yaml


def main(config):
    
    pl.seed_everything(config['random_seed'], workers=True)
    rtbwe_datamodule = RTBWEDataModule(config)
    rtbwe_train = RTBWETrain(config)
    
    check_dir_exist(config['train']['output_dir_path'])
    check_dir_exist(config['train']['logger_path'])
    
    tb_logger = pl_loggers.TensorBoardLogger(config['train']['logger_path'], name=f"RTBWE_logs")
    
    checkpoint_callback = ModelCheckpoint(
    filename = "{epoch}-{val_pesq_wb:.2f}-{val_pesq_nb:.2f}",
    save_top_k = -1,
    every_n_epochs = config['train']['val_epoch'])

    tb_logger.log_hyperparams(config)

    trainer=pl.Trainer(devices=config['train']['devices'], accelerator="gpu", strategy='ddp_find_unused_parameters_true',
    callbacks= [checkpoint_callback],
    max_epochs=config['train']['max_epochs'],
    logger=tb_logger,
    check_val_every_n_epoch=config['train']['val_epoch']
    )
    
    trainer.fit(rtbwe_train, rtbwe_datamodule)
    trainer.save_checkpoint(os.path.join(config['train']['output_dir_path'],'final_model.ckpt'))
    
if __name__ == "__main__":
    config = yaml.load(open("./config.yaml", 'r'), Loader=yaml.FullLoader)
    
    main(config)