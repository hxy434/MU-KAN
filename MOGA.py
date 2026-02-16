import torch
import torch.nn as nn
import torch.nn.functional as F



class ElementScale(nn.Module):
    
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__() 

        
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),  
            requires_grad=requires_grad  
        )
      

    def forward(self, x):
        return x * self.scale  


class MultiOrderDWConv(nn.Module):

    
    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3],
                 channel_split=[1, 3, 4],
                ):
        super(MultiOrderDWConv, self).__init__()

        
        self.split_ratio = [i / sum(channel_split) for i in channel_split]

        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)  
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)  
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2 

        self.embed_dims = embed_dims

      
        assert len(dw_dilation) == len(channel_split) == 3  
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3  
        assert embed_dims % sum(channel_split) == 0  

        
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2, 
            groups=self.embed_dims,               
            stride=1,
            dilation=dw_dilation[0],              
        )
       
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1,
            dilation=dw_dilation[1],  
        )
        
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1,
            dilation=dw_dilation[2], 
        )
       
        self.PW_conv = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1
        )

    def forward(self, x):
        x_0 = self.DW_conv0(x)  

        
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])

        
        x_2 = self.DW_conv2(x_0[:, self.embed_dims - self.embed_dims_2:, ...])

      
        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)

        x = self.PW_conv(x)  
        return x


class MultiOrderGatedAggregation(nn.Module):
  
    
    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_force_fp32=False,
                 ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
       
        self.proj_1 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
     
        self.gate = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )

      
        self.proj_2 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

     
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

        self.sigma = ElementScale(embed_dims, init_value=1e-5, requires_grad=True)
      
    def feat_decompose(self, x):
     
        x = self.proj_1(x)

       
        x_d = F.adaptive_avg_pool2d(x, output_size=1)

        
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward(self, x):
        shortcut = x.clone()  

    
        x = self.feat_decompose(x)

        
        F_branch = self.gate(x)
        G_branch = self.value(x)

       
        x = self.proj_2(self.act_gate(F_branch) * self.act_gate(G_branch))
        x = x + shortcut  
       
        return x

if __name__ == '__main__':

    input = torch.randn(1, 64, 32, 32)

    MOGA = MultiOrderGatedAggregation(64) 

    output = MOGA(input)
    print('MOGA_input_size:', input.size())
    print('MOGA_output_size:', output.size())
