import torch
import torch as th
import torch.nn as nn

class Discriminator_1D(nn.Module):
    def __init__(self, bias= True):
        
        super().__init__()
        self.conv1 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 1, out_channels = 16, 
                                          kernel_size = 15, stride= 1, padding= 7, bias= bias)),
                                nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 16*1, out_channels = 64, 
                                          kernel_size = 41, stride= 4, groups = 4, padding = 20, bias= bias)),
                                nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 64, out_channels = 256, 
                                          kernel_size = 41, stride= 4, groups = 4, padding = 20, bias= bias)),
                                nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 256, out_channels = 1024, 
                                          kernel_size = 41, stride= 4, groups = 4, padding = 20, bias= bias)),
                                nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 1024, out_channels = 1024, 
                                          kernel_size = 41, stride= 4, groups = 4, padding = 20, bias= bias)),
                                nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 1024, out_channels = 1024, 
                                          kernel_size = 5, stride= 1, groups = 1, padding = 2, bias= bias)),
                                nn.LeakyReLU(0.2))
        self.conv7 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 1024, out_channels = 1, 
                                          kernel_size = 3, stride= 1, groups = 1, padding = 1, bias= bias)))
        
        
    def forward(self, x):
        while len(x.size()) <= 2:
            x = x.unsqueeze(-2)                
        xs = []
        x = self.conv1(x)
        xs.append(x)
        x = self.conv2(x)
        xs.append(x)
        x = self.conv3(x)
        xs.append(x)
        x = self.conv4(x)
        xs.append(x)
        x = self.conv5(x)
        xs.append(x)
        x = self.conv6(x)
        xs.append(x)
        x = self.conv7(x)
        xs.append(x)
        return x, xs

class Discriminator_MelGAN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.disc = nn.ModuleList([Discriminator_1D(bias = True) for i in range(3)])
        self.pool1 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)
        self.pool2 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)
        
    def forward(self, x):
        
        while len(x.size()) <= 2:
            x = x.unsqueeze(-2)     

        x1 = x
        x2 = self.pool1(x1)
        x3 = self.pool2(x2)
        
        d1, f1 = self.disc[0](x1)
        d2, f2 = self.disc[1](x2)
        d3, f3 = self.disc[2](x3)
        return (d1, d2, d3), (f1, f2, f3)

    def loss_D(self, x_proc, x_orig, *args, **kwargs):
        x_proc = x_proc.squeeze()[...,:x_orig.shape[-1]].detach()
        x_orig = x_orig.squeeze()[...,:x_proc.shape[-1]]
        
        D_proc, F_proc = self(x_proc)
        D_orig, F_orig = self(x_orig)

        loss = 0

        loss_GAN  = []
        for r in range(len(D_proc)):
            dist = (1-D_orig[r]).relu().mean() + (1+D_proc[r]).relu().mean()  # Hinge loss
            
            loss_GAN.append(dist)
        loss_GAN = sum(loss_GAN)/len(loss_GAN)
        
        loss += loss_GAN

        return loss

    def loss_G(self, x_proc, x_orig, *args, **kwargs):
        x_proc = x_proc.squeeze()[...,:x_orig.shape[-1]]
        x_orig = x_orig.squeeze()[...,:x_proc.shape[-1]]

        D_proc, F_proc = self(x_proc)
        D_orig, F_orig = self(x_orig)

        loss_GAN  = []
        loss_FM = []
        

        for r in range(len(D_proc)):

            loss_GAN.append((1-D_proc[r]).relu().mean())

            for l in range(len(F_proc[r])-1):
                loss_FM.append((F_proc[r][l] - F_orig[r][l].detach()).abs().mean())
                    
        loss_GAN  = sum(loss_GAN)/len(loss_GAN) 
        loss_FM = sum(loss_FM)/len(loss_FM)  

        loss = 100*loss_FM + loss_GAN
        return loss
    
    def get_name(self):
        return self.name

if __name__ == "__main__":
    audio = torch.rand(4,1,64000)
    noisy = torch.rand(4,1,64000)

    melgan = Discriminator_MelGAN()
    print(melgan.loss_G(audio, noisy))