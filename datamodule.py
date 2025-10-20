import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio as ta
import numpy as np
import os

import pytorch_lightning as pl


from utils import *


class RTBWEDataset(Dataset): 
  #데이터셋의 전처리를 해주는 부분
    def __init__(self, path_dir_nb, path_dir_wb, seg_len, mode="train"):
        self.path_dir_nb   = path_dir_nb
        self.path_dir_wb   = path_dir_wb  

        self.seg_len = seg_len
        self.mode = mode

        self.wavs={}
        self.filenames= []

        paths_wav_wb= get_wav_paths(self.path_dir_wb)
        paths_wav_nb= get_wav_paths(self.path_dir_nb)

        if mode == "pred":
            for path_wav_nb in paths_wav_nb:
                filename=get_filename(path_wav_nb)[0]

                wav_nb, self.sr_nb = ta.load(path_wav_nb)
                
                if self.sr_nb != 8000:
                    wav_nb = ta.functional.resample(wav_nb, self.sr_nb, 8000)
                
                self.wavs[filename]=(None , wav_nb)
                self.filenames.append(filename)
                print(f'\r {mode}: {len(self.filenames)} th file loaded', end='')
        
        else:
            for path_wav_wb, path_wav_nb in zip(paths_wav_wb, paths_wav_nb):
                filename=get_filename(path_wav_wb)[0]
                wav_nb, self.sr_nb = ta.load(path_wav_nb)
                wav_wb, self.sr_wb = ta.load(path_wav_wb)
                
                if self.sr_nb != 8000:
                    wav_nb = ta.functional.resample(wav_nb, self.sr_nb, 8000)
                if self.sr_wb != 16000:
                    wav_wb = ta.functional.resample(wav_wb, self.sr_wb, 16000)
                
                self.wavs[filename]=(wav_wb, wav_nb)
                self.filenames.append(filename)
                print(f'\r {mode}: {len(self.filenames)} th file loaded', end='')
            
        self.filenames.sort()


       
    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.filenames)


    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):

        filename = self.filenames[idx]
        (wav_wb, wav_nb) = self.wavs[filename]


        if self.seg_len > 0 and self.mode == "train":
            duration= int(self.seg_len * 16000)

            wav_wb= wav_wb.view(1,-1)
            wav_nb = wav_nb.view(1,-1)

            sig_len = wav_wb.shape[-1]

            t_start = np.random.randint(
                low = 0,
                high= np.max([1, sig_len- duration - 2]),
                size = 1
            )[0]

            if t_start % 2 ==1:
                t_start -= 1

            t_end = t_start + duration


            wav_nb = wav_nb.repeat(1, t_end// sig_len + 1) [ ..., t_start//2 : t_end//2]
            wav_wb = wav_wb.repeat(1, t_end // sig_len + 1) [ ..., t_start : t_end]
        else:
            wav_wb= wav_wb.view(1,-1)
            wav_nb = wav_nb.view(1,-1)
            
            #wav_nb padding
            nb_padding = 256 - len(wav_nb[-1])%256
            wav_nb = torch.cat([wav_nb, torch.zeros((1, nb_padding))], dim=1)
            
            #wav_wb padding
            wb_len = wav_nb.shape[1]*2
            wb_padding = wb_len - len(wav_wb[-1])
            wav_wb = torch.cat([wav_wb, torch.zeros((1, wb_padding))], dim=1)
            
            
        return wav_nb, wav_wb, filename


class RTBWEDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        
        self.data_dir = config['dataset']['data_dir']
        
        self.path_dir_nb_train = config['dataset']['nb_train']
        self.path_dir_nb_val = config['dataset']['nb_val']
        self.path_dir_wb_train =  config['dataset']['wb_train']
        self.path_dir_wb_val =  config['dataset']['wb_val']
        
        self.path_dir_nb_pred = config['predict']['nb_pred_path']

        
        
        self.batch_size = config['dataset']['batch_size']
        self.seg_len = config['dataset']['seg_len']
        
        self.num_workers = config['dataset']['num_workers']


    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        self.train_dataset =RTBWEDataset(
            path_dir_nb = os.path.join(self.data_dir, self.path_dir_nb_train),
            path_dir_wb = os.path.join(self.data_dir, self.path_dir_wb_train),
            seg_len = self.seg_len,
            mode = "train"
            )

        self.val_dataset = RTBWEDataset(
            path_dir_nb = os.path.join(self.data_dir, self.path_dir_nb_val),
            path_dir_wb = os.path.join(self.data_dir, self.path_dir_wb_val),
            seg_len = self.seg_len,
            mode = "val"
            )

         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = 1, num_workers = self.num_workers)
