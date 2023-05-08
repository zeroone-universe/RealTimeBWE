import argparse
import torchaudio as ta
import pytorch_lightning as pl

from train import RTBWETrain
from datamodule import *
from utils import *
import yaml

def inference(config, args):

    rtbwe_train = RTBWETrain.load_from_checkpoint(args.path_ckpt, config = config)
    
    if args.mode == 'wav':
        wav_nb, sr_nb = ta.load(args.path_in)
        wav_nb = wav_nb.unsqueeze(0)
        rtbwe_train.generator.eval()
        wav_bwe = rtbwe_train.forward(wav_nb)
        
        filename = get_filename(args.path_in)
        ta.save(os.path.join(os.path.dirname(args.path_in),filename[0]+"_bwe"+filename[1]), wav_bwe.squeeze(0), sr_nb*2)
        
    if args.mode == 'dir':
        
        pred_dataset = RTBWEDataset(
            path_dir_nb = config["predict"]["nb_pred_path"],
            path_dir_wb = config["predict"]["nb_pred_path"],
            mode = "pred"
        )
        trainer = pl.Trainer(devices=1, accelerator="gpu", logger = False)
        
        trainer.predict(rtbwe_train, pred_dataset)
    
        

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, help = "wav/dir", default = "wav")
    parser.add_argument("--path_ckpt", type = str)
    parser.add_argument("--path_in", type = str, help = "path of wav file or directory")
    args = parser.parse_args()
    
    config = yaml.load(open("./config.yaml", 'r'), Loader=yaml.FullLoader)
    
    inference(config, args)

