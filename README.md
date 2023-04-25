# Real-time Speech Frequency Bandwidth Extension

This repository contains the unofficial pytorch lightning implementation of the model described in the paper [Real-Time Speech Frequency Bandwidth Extension](https://arxiv.org/pdf/2010.10677.pdf) by Yunpeng Li et al. (2021). 

## Requirements
 
To run this code, you will need:

- torch==2.0.0
- pytorch_lightning==2.0.0
- numpy==1.23.5
- pesq==0.0.4
- PyYAML==6.0
- torchaudio==2.0.0

To automatically install these libraries, run the following command:

```pip install -r requirements.txt```

## Usage

To run the code on your own machine, follow these steps:

1. Open the 'config.yaml' file and modify the file paths (and hyperparameters as needed).
2. Run the 'main.py' file to start training the model.

The trained model will be saved as ckpt file in 'logger' directory. You can then use the trained model to perform real-time speech frequency bandwidth extension on your own audio wav file by running the 'inference.py' file as

```python inference.py --mode wav --path_ckpt <path of checkpoint file> --path_in <path of wav file>```

This repository also support directory-level inference, where the inference is performed on a directory consisting of wav files. Before running directory-level inference, it is necessary to modify the 'predict' section of the config.yaml file. You can use the following example to perform directory-level inference,

```python inference.py --mode dir --path_ckpt <path of checkpoint file>```

## Note
- This implementation does not include streaming convolution and uses the conventional causal convolution instead. Although this deviates from the contributions of the original paper, I am focusing on verifying the bandwidth extension performance of this model.
- The original paper conducted training for 1 million steps, whereas this implementation trained for 350 epochs for personal research convenience. The number of epochs can be adjusted arbitrarily. 
- Feel free to provide issues!

## Citation

```bibtex
@inproceedings{SEANetBWE21,
  title={Real-time speech frequency bandwidth extension},
  author={Li, Yunpeng and Tagliasacchi, Marco and Rybakov, Oleg and Ungureanu, Victor and Roblek, Dominik},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={691--695},
  year={2021},
  organization={IEEE}
}
```

