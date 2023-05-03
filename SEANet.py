import torch as th
import torch 
import torch.nn as nn
import torch.nn.functional as F


class SEANet(nn.Module):
    def __init__(self, min_dim=8, **kwargs):
        super().__init__()
        
        self.min_dim = min_dim
        
        self.conv_in = Conv1d(
            in_channels = 1,
            out_channels = min_dim,
            kernel_size = 7,
            stride = 1
        )
        
        self.encoder = nn.ModuleList([
                                    EncBlock(min_dim*2, 2),
                                    EncBlock(min_dim*4, 2),
                                    EncBlock(min_dim*8, 8),
                                    EncBlock(min_dim*16, 8)                                        
                                    ])
        
        self.conv_bottle = nn.Sequential(
                                        Conv1d(
                                            in_channels=min_dim*16,
                                            out_channels = min_dim*16//4,
                                            kernel_size = 7, 
                                            stride = 1,
                                            ),
                                        
                                        Conv1d(
                                            in_channels=min_dim*16//4,
                                            out_channels = min_dim*16,
                                            kernel_size = 7,
                                            stride = 1,
                                            ),
                                        )
        
        self.decoder = nn.ModuleList([
                                    DecBlock(min_dim*8, 8),
                                    DecBlock(min_dim*4, 8),
                                    DecBlock(min_dim*2, 2),
                                    DecBlock(min_dim, 2),
                                    ])
        
        self.conv_out = Conv1d(
            in_channels = min_dim,
            out_channels = 1,
            kernel_size = 7,
            stride = 1,
        )
        
    def forward(self, x):

        while len(x.size()) < 3:
            x = x.unsqueeze(-2)

        skip = [x]
        
        x = self.conv_in(x)
        skip.append(x)

        for encoder in self.encoder:
            x = encoder(x)
            skip.append(x)

        x = self.conv_bottle(x)

        skip = skip[::-1]

        for l in range(len(self.decoder)):
            x = x + skip[l]
            x = self.decoder[l](x)

        x = x + skip[4]
        x = self.conv_out(x)
        
        x = x + skip[5]
        return x
    
    
class EncBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()
        

        self.res_units = nn.ModuleList([
                                    ResUnit(out_channels//2, 1),
                                    ResUnit(out_channels//2, 3),
                                    ResUnit(out_channels//2, 9)                                        
                                    ])
        
        self.conv = nn.Sequential(
                    nn.ELU(),
                    Pad((2 * stride - 1, 0)),
                    nn.Conv1d(in_channels = out_channels//2,
                                       out_channels = out_channels,
                                       kernel_size = 2 * stride,
                                       stride = stride, padding = 0),
                    )  
        
        
    def forward(self, x):
        
        for res_unit in self.res_units:
            x = res_unit(x)
        x = self.conv(x)

        return x
        
    
class DecBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        
        self.conv = ConvTransposed1d(
                                 in_channels = out_channels*2, 
                                 out_channels = out_channels, 
                                 kernel_size = 2*stride, stride= stride,
                                 dilation = 1,
                                 )
        
        
        self.res_units = nn.ModuleList([
                                    ResUnit(out_channels, 1),
                                    ResUnit(out_channels, 3),
                                    ResUnit(out_channels, 9)                                       
                                    ])
               
        self.stride = stride
        

    def forward(self, x):
        x = self.conv(x)
        for res_unit in self.res_units:
            x = res_unit(x)
        return x
    
    
class ResUnit(nn.Module):
    def __init__(self, channels, dilation = 1):
        super().__init__()
        

        self.conv_in = Conv1d(
                                 in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 3, stride= 1,
                                 dilation = dilation,
                                 )
        
        self.conv_out = Conv1d(
                                in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 )
        
        self.conv_shortcuts = Conv1d(
                                in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 )
        
    
        
    def forward(self, x):
        y = self.conv_in(x)
        y = self.conv_out(y)
        x = self.conv_shortcuts(x)
        return x + y
        
    
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation = 1, groups = 1):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels = in_channels, 
            out_channels = out_channels,
            kernel_size= kernel_size, 
            stride= stride, 
            dilation = dilation,
            groups = groups
        )
        self.conv = nn.utils.weight_norm(self.conv)
        
        self.pad = Pad(((kernel_size-1)*dilation, 0)) 
        self.activation = nn.ELU()
            

    def forward(self, x):

        x = self.pad(x)
        x = self.conv(x)
        x = self.activation(x)
        
        return x

class ConvTransposed1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, dilation = 1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride =stride,
            dilation = dilation
        )
        self.conv = nn.utils.weight_norm(self.conv)
        
        self.activation = nn.ELU()
        self.pad = dilation * (kernel_size - 1) - dilation * (stride - 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = x[..., :-self.pad]
        x = self.activation(x)
        return x
    
class Pad(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    
    def forward(self, x):
        return F.pad(x, pad=self.pad)    
    
    
if __name__ == "__main__":

    model = SEANet(in_channels=1, out_channels=1, min_dim = 32)
    wav = torch.rand(4,1,55400)
    output = model(wav)
    print(f"output_shape: {output.shape}")