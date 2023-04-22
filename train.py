import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

import torchaudio as ta

import os

from MelGAN import Discriminator_MelGAN
from SEANet import SEANet


from utils import *

from pesq import pesq

class RTBWETrain(pl.LightningModule):
    def __init__(self, config):
        super(RTBWETrain, self).__init__()
        self.config = config

        self.lr = config['optim']['learning_rate']
        self.B1 = config['optim']['B1']
        self.B2 = config['optim']['B2']

        self.generator = SEANet(min_dim = 8, causality = True, skip_outmost = True)
        self.output_dir_path = config['train']['output_dir_path']

        self.epoch_save_start = config['train']['epoch_save_start']
        self.val_epoch = config['train']['val_epoch']
        
        self.path_dir_bwe_pred =  config['predict']['pred_output_path']
        
        self.discriminator = Discriminator_MelGAN()

        self.resampler = T.Resample(8000, 16000, dtype = torch.float32)
        
        self.automatic_optimization = False

    def forward(self,x):
        x = self.resampler(x)
        output = self.generator(x)

        return output

    def configure_optimizers(self):
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas = (self.B1, self.B2))
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas = (self.B1, self.B2))

        return optimizer_d, optimizer_g #, [lrscheduler_d, lr_scheduler_g])


    def training_step(self, batch, batch_idx):
        optimizer_d, optimizer_g = self.optimizers()
        
        wav_nb, wav_wb, _ = batch
        
            
        wav_bwe = self.forward(wav_nb)

        #optimize discriminator
        
        self.toggle_optimizer(optimizer_d)
        
        loss_d =self.discriminator.loss_D(wav_bwe, wav_wb)
        
        optimizer_d.zero_grad()
        self.manual_backward(loss_d)
        optimizer_d.step()

        self.untoggle_optimizer(optimizer_d)
        
        #optimize generator

        self.toggle_optimizer(optimizer_g)

        loss_g = self.discriminator.loss_G(wav_bwe, wav_wb)
        
        optimizer_g.zero_grad()
        self.manual_backward(loss_g)
        optimizer_g.step()
        
        self.untoggle_optimizer(optimizer_g)
        
        self.log("train_loss_d", loss_d, prog_bar = True, batch_size = self.config['dataset']['batch_size'])
        self.log("train_loss_g", loss_g, prog_bar = True, batch_size = self.config['dataset']['batch_size'])

            

    def validation_step(self, batch, batch_idx):
        
        wav_nb, wav_wb, filename = batch
    
        wav_bwe = self.forward(wav_nb)
        
        
        loss_d = self.discriminator.loss_D(wav_bwe, wav_wb)
        loss_g = self.discriminator.loss_G(wav_bwe, wav_wb)
       
       
        wav_bwe_cpu = wav_bwe.squeeze(0).cpu()
        val_dir_path = f"{self.output_dir_path}/epoch_current"
        check_dir_exist(val_dir_path)
        ta.save(os.path.join(val_dir_path, f"{filename[0]}.wav"), wav_bwe_cpu, 16000)

        wav_wb = wav_wb.squeeze().cpu().numpy()
        wav_bwe = wav_bwe.squeeze().cpu().numpy()

        val_pesq_wb = pesq(fs = 16000, ref = wav_wb, deg = wav_bwe, mode = "wb")
        val_pesq_nb = pesq(fs = 16000, ref = wav_wb, deg = wav_bwe, mode = "nb")
        
        self.log_dict({"val_loss/val_loss_d": loss_d, "val_loss/val_loss_g": loss_g}, batch_size = 1)
        self.log('val_pesq_wb', val_pesq_wb, batch_size = 1)
        self.log('val_pesq_nb', val_pesq_nb, batch_size = 1)


    def test_step(self, batch, batch_idx):
        pass



    def predict_step(self, batch, batch_idx):
        wav_nb, _, filename = batch

        wav_bwe = self.forward(wav_nb)
        
        wav_bwe_cpu = wav_bwe.squeeze(0).cpu()
        test_dir_path = self.path_dir_bwe_pred
        check_dir_exist(test_dir_path)
        ta.save(os.path.join(test_dir_path, f"{filename}.wav"), wav_bwe_cpu, 16000)
