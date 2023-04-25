import torch as th
import torch 
import torch.nn as nn
import torch.nn.functional as F


class SEANet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, min_dim=32, causal = True, weight_norm = True, skip_connection = True, skip_outmost = True, **kwargs):
        super().__init__()
        
        self.name = 'SEANet_%s'%('cau' if causal else 'ncau')
        
        self.min_dim = min_dim
        self.causal = causal
        self.weight_norm = weight_norm
        self.skip_connection = skip_connection
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.skip_outmost = skip_outmost
        
        self.conv_in = Conv1d(in_channels = in_channels,
                                 out_channels = min_dim,
                                 kernel_size = 7,
                                 stride = 1,
                                 bias = False, 
                                 activation = None, causality = causal, pre_activation=False)
        
        self.encoder = nn.ModuleList([
                                    EncBlock(min_dim*2, 2, weight_norm, causal),
                                    EncBlock(min_dim*4, 2, weight_norm, causal),
                                    EncBlock(min_dim*8, 8, weight_norm, causal),
                                    EncBlock(min_dim*16, 8, weight_norm, causal)                                        
                                    ])
        
        self.conv_bottle = nn.Sequential(
                                        Conv1d(in_channels=min_dim*16,
                                                  out_channels = min_dim*16//4,
                                                  kernel_size = 7, stride = 1,
                                                  activation = 'ELU', causality= causal, pre_activation = True),
                                        
                                        Conv1d(in_channels=min_dim*16//4,
                                                  out_channels = min_dim*16,
                                                  kernel_size = 7, stride = 1,
                                                  activation = 'ELU', causality= causal, pre_activation = True),
                                        )
        
        self.decoder = nn.ModuleList([
                                    DecBlock(min_dim*8, 8, weight_norm, causal),
                                    DecBlock(min_dim*4, 8, weight_norm, causal),
                                    DecBlock(min_dim*2, 2, weight_norm, causal),
                                    DecBlock(min_dim, 2, weight_norm, causal),
                                    ])
        
        self.conv_out = Conv1d(in_channels = min_dim,
                                   out_channels = out_channels,
                                   kernel_size = 7,
                                   stride = 1,
                                   bias= False,
                                   activation = None, causality = causal, pre_activation=False)
        
    def forward(self, x):

        while len(x.size()) < 3:
            x = x.unsqueeze(-2)
        
        if self.skip_connection:
            y = [x]
        
        x = self.conv_in(x)
        if self.skip_connection:
            y.append(x)

        for encoder in self.encoder:
            # print(x.shape)
            x = encoder(x)
            # print(x.shape)
            if self.skip_connection:
                y.append(x)
        # print(x.shape)

        x = self.conv_bottle(x)
        # print(x.shape)

        for l in range(len(self.decoder)):
            # print(x.shape)
            if self.skip_connection:
                x = x[..., :y[-l-1].shape[-1]] + y[-l-1][..., :x.shape[-1]]
            # print(x.shape)
            
            x = self.decoder[l](x)
            
            # print(x.shape)
        if self.skip_connection:
            x = x[..., :y[1].shape[-1]] + y[1][..., :x.shape[-1]]
            # print(x.shape)
        x = self.conv_out(x)

        if self.skip_outmost:
            x = x[..., :y[0].shape[-1]] + y[0][..., :x.shape[-1]]
        
        return x
    def get_name(self):
        return self.name
    
class EncBlock(nn.Module):
    def __init__(self, out_channels, stride, weight_norm, causality):
        super().__init__()
        

        self.res_units = nn.ModuleList([
                                    ResUnit(out_channels//2, 1, weight_norm, causality),
                                    ResUnit(out_channels//2, 3, weight_norm, causality),
                                    ResUnit(out_channels//2, 9, weight_norm, causality)                                        
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
    def __init__(self, out_channels, stride, weight_norm, causality):
        super().__init__()

        
        self.conv = ConvTransposed1d(
                                 in_channels = out_channels*2, 
                                 out_channels = out_channels, 
                                 kernel_size = 2*stride, stride= stride,
                                 dilation = 1,
                                 activation = None,
                                 causality = causality,
                                 bias= True
                                 )
        
        
        self.res_units = nn.ModuleList([
                                    ResUnit(out_channels, 1, weight_norm, causality),
                                    ResUnit(out_channels, 3, weight_norm, causality),
                                    ResUnit(out_channels, 9, weight_norm, causality)                                       
                                    ])
               
        self.stride = stride
        

    def forward(self, x):
        x = self.conv(x)
        for res_unit in self.res_units:
            x = res_unit(x)
        return x
    
    
class ResUnit(nn.Module):
    def __init__(self, channels, dilation = 1, weight_norm=True, causality=True):
        super().__init__()
        

        self.conv_in = Conv1d(
                                 in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 3, stride= 1,
                                 dilation = dilation,
                                 activation = 'ELU',
                                 weight_norm = weight_norm,
                                 causality = causality, pre_activation=True
                                 )
        
        self.conv_out = Conv1d(
                                in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 activation = 'ELU',
                                 weight_norm = weight_norm,
                                 causality = causality, pre_activation=True
                                 )
        
        self.conv_shortcuts = Conv1d(
                                in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 activation = 'ELU',
                                 weight_norm = weight_norm,
                                 causality = causality, pre_activation=True
                                 )
        
    
        
    def forward(self, x):
        y = self.conv_in(x)
 
        y = self.conv_out(y)
        x = self.conv_shortcuts(x)
        return x + y
        
    
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 dilation = 1, groups = 1, bias= False, 
                 causality=False, weight_norm=True, activation='ReLU', pre_activation = False):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels = in_channels, out_channels = out_channels,
                              kernel_size= kernel_size, stride= stride, 
                              dilation = dilation, groups = groups, bias= bias)
        
        self.pad = Pad(((kernel_size-1) * dilation, 0)) if causality else Pad((((kernel_size-1)*dilation)//2, ((kernel_size-1)*dilation)//2))
        
        self.activation = getattr(nn, activation)() if activation != None else Identity()
            
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        self.pre_activation = pre_activation
        
    def forward(self, x):
        
        if self.pre_activation:
            x = self.activation(x)
        
        x = self.pad(x)
        x = self.conv(x)
        
        if not self.pre_activation:
            x = self.activation(x)
        return x

class ConvTransposed1d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size = 1, stride = 1, dilation = 1,
                causality = True, bias=False, activation = 'ELU'):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride =stride,
                              dilation = dilation,
                              bias = bias)
        
        self.activation = getattr(nn, activation)() if activation is not None else None
        self.pad = dilation * (kernel_size - 1) - dilation * (stride - 1) if causality else (dilation * (kernel_size - 1))//2
        self.causality = causality
        
    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        x = x[..., :-self.pad] if self.causality else x[..., self.pad:]
        return x
                    
        
    

class Identity(nn.Module):
    def __init__(self, opt_print=False):
        super().__init__()
        self.opt_print = opt_print
    def forward(self, x):
        if self.opt_print: print(x.shape)
        return x
    
class Pad(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    
    def forward(self, x):
        return F.pad(x, pad=self.pad)    
    
    
if __name__ == "__main__":

    model = SEANet(in_channels=1, out_channels=1, min_dim = 32, causal=True, weight_norm = True, skip_connection = True, skip_outmost = True)

    
    test_model(model)
   