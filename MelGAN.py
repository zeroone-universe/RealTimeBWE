import torch
import torch as th
import torch.nn as nn

class Discriminator_1D(nn.Module):
    def __init__(self, 
                 first_in_channels = 1, 
                 first_stride=1, 
                 last_out_channels = 1,
                 bias= True):
        super().__init__()
        self.conv1 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = first_in_channels, out_channels = 16*first_stride, 
                                          kernel_size = 15*first_stride, stride= first_stride, padding= 15*first_stride//2, bias= bias)),
                                nn.LeakyReLU(0.3))
        self.conv2 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 16*first_stride, out_channels = 64, 
                                          kernel_size = 41, stride= 4, groups = 4, padding = 20, bias= bias)),
                                nn.LeakyReLU(0.3))
        self.conv3 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 64, out_channels = 256, 
                                          kernel_size = 41, stride= 4, groups = 16, padding = 20, bias= bias)),
                                nn.LeakyReLU(0.3))
        self.conv4 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 256, out_channels = 1024, 
                                          kernel_size = 41, stride= 4, groups = 64, padding = 20, bias= bias)),
                                nn.LeakyReLU(0.3))
        self.conv5 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 1024, out_channels = 1024, 
                                          kernel_size = 41, stride= 4, groups = 256, padding = 20, bias= bias)),
                                nn.LeakyReLU(0.3))
        self.conv6 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 1024, out_channels = 1024, 
                                          kernel_size = 5, stride= 1, groups = 1, padding = 2, bias= bias)),
                                nn.LeakyReLU(0.3))
        self.conv7 = nn.Sequential(
                                th.nn.utils.weight_norm(nn.Conv1d(in_channels = 1024, out_channels = last_out_channels, 
                                          kernel_size = 3, stride= 1, groups = 1, padding = 1, bias= bias)))
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         th.nn.utils.weight_norm(m)        
        
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
    def __init__(self, D_adv = True, D_fm = False, G_adv= True, G_fm=True,  **kwargs):
        super().__init__()
        
        self.name = 'MelGAN{D_adv}{D_fm}{G_adv}{G_fm}'.format(
                    D_adv = '_Dadv' if D_adv else '',
                    D_fm = '_Dfm' if D_fm else '',
                    G_adv = '_Gadv' if G_adv else '',
                    G_fm = '_Gfm' if G_fm else '',
                    )
        
        self.D_adv = D_adv
        self.D_fm  = D_fm
        self.G_adv = G_adv
        self.G_fm  = G_fm
        
        self.disc = nn.ModuleList([Discriminator_1D(bias=False) for i in range(3)])
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
        
        # if self.normalize:
        #     x_proc = x_proc/(x_proc.std(dim=-1, keepdim=True)+1e-10)
        #     x_orig = x_orig/(x_orig.std(dim=-1, keepdim=True)+1e-10)
        
        D_proc, F_proc = self(x_proc)
        D_orig, F_orig = self(x_orig)

        loss = 0
        if self.D_adv:
            loss_GAN  = []
            for r in range(len(D_proc)):
                dist = (1-D_orig[r]).relu().mean() + (1+D_proc[r]).relu().mean()  # Hinge loss
                # dist = ( D_orig[r]/(D_orig[r].abs().mean(dim=-1, keepdim=True)+1e-10) - D_proc[r]/(D_proc[r].abs().mean(dim=-1, keepdim=True)+1e-10)).mean()   # WGAN loss
                
                loss_GAN.append(dist)
            loss_GAN = sum(loss_GAN)/len(loss_GAN)
            
            loss += loss_GAN

        if self.D_fm:
            loss_FM = [] 
            for r in range(len(D_proc)):
                # loss_FM.append(D_proc[r].mean()-D_orig[r].mean())
                dist_nomin = (D_proc[r]-D_orig[r]).abs()
                dist_denom = ((D_proc[r]*D_orig[r]).abs()+1e-10).sqrt()
                dist = dist_nomin / dist_denom

                # dist_norm = dist/(D_proc[r].abs()+D_orig[r].abs()+1e-10)
                # dist_norm = dist/(D_proc[r].abs().mean(dim=-1, keepdim=True)+D_orig[r].abs().mean(dim=-1, keepdim=True)+1e-10)
                # loss_FM.append((1-dist_norm).relu())


                loss_FM.append(-dist.mean())

            loss_FM = sum(loss_FM)/len(loss_FM)
            
            loss += loss_FM
        return loss

    def loss_G(self, x_proc, x_orig, *args, **kwargs):
        x_proc = x_proc.squeeze()[...,:x_orig.shape[-1]]
        x_orig = x_orig.squeeze()[...,:x_proc.shape[-1]]
        
        # if self.normalize:
        #     x_orig = x_orig/(x_orig.std(dim=-1, keepdim=True)+1e-10)
        #     x_proc = x_proc/(x_proc.std(dim=-1, keepdim=True)+1e-10)     
        
        D_proc, F_proc = self(x_proc)
        D_orig, F_orig = self(x_orig)



        loss_GAN  = []
        loss_FM = []
        

        for r in range(len(D_proc)):
            
            if self.G_adv:
                loss_GAN.append((1-D_proc[r]).relu().mean())
                # loss_GAN.append((D_proc[r]/(D_proc[r].abs().mean(dim=-1, keepdim=True)+1e-10)).mean())

            if self.G_fm:

                for l in range(len(F_proc[r])-1):
                    loss_FM.append((F_proc[r][l] - F_orig[r][l].detach()).abs().mean())
                    # loss_FM.append((F_proc[r][l] - F_orig[r][l]).abs().mean())
            
                    
        loss_GAN  = sum(loss_GAN)/len(loss_GAN) if self.G_adv else 0
        loss_FM = sum(loss_FM)/len(loss_FM)  if self.G_fm else 0

        loss = 100*loss_FM + loss_GAN
        return loss
    
    def get_name(self):
        return self.name

if __name__ == "__main__":
    audio = torch.rand(4,1,64000)
    noisy = torch.rand(4,1,64000)

    melgan = Discriminator_MelGAN()
    print(melgan.loss_G(audio, noisy))