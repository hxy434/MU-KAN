import torch
import torch.nn as nn
import torch.nn.functional as F

"""
论文链接：https://openreview.net/pdf?id=XhYWgjqCrV
文章标题：MOGANET: MULTI-ORDER GATED AGGREGATION NETWORK（ICLR 2024）
"""

class ElementScale(nn.Module):
    
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()  # 调用父类构造函数初始化模块

        # 初始化一个可训练的缩放参数self.scale，用于对输入张量进行逐元素加权。
        # 例如，当embed_dims为64且初始值为0.5时，self.scale的形状为(1, 64, 1, 1)，所有值均为0.5。
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),  # 初始缩放因子
            requires_grad=requires_grad  # 决定该参数是否参与梯度更新
        )
        # ai缝合大王

    def forward(self, x):
        return x * self.scale  # 返回输入与缩放因子相乘的结果


class MultiOrderDWConv(nn.Module):
    """基于膨胀深度卷积实现多阶特征提取模块。

    参数:
        embed_dims (int): 输入特征图的通道数。
        dw_dilation (list): 三个深度卷积层对应的膨胀因子。
        channel_split (list): 不同分支的通道分配比例。
    """
    
    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3],
                 channel_split=[1, 3, 4],
                ):
        super(MultiOrderDWConv, self).__init__()

        # 根据channel_split计算每个分支的通道比例，例如：1/8, 3/8, 4/8
        self.split_ratio = [i / sum(channel_split) for i in channel_split]

        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)  # 第二部分通道数：约3/8 * embed_dims
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)  # 第三部分通道数：约4/8 * embed_dims
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2  # 第一部分通道数：剩余部分

        self.embed_dims = embed_dims

        # 检查参数长度和合理性：确保dw_dilation和channel_split均为3个元素，并且膨胀率在1到3之间，同时embed_dims能被分割比例整除
        assert len(dw_dilation) == len(channel_split) == 3  
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3  
        assert embed_dims % sum(channel_split) == 0  

        # 基础深度卷积：对整个输入进行5x5卷积操作，填充值依据膨胀率计算
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,  # 根据膨胀因子计算所需填充
            groups=self.embed_dims,                # 分组数等于通道数，实现深度卷积
            stride=1,
            dilation=dw_dilation[0],               # 膨胀率为dw_dilation[0]（通常为1）
        )
        # 第二个深度卷积：对第一分支（通道占比约3/8）应用5x5卷积
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1,
            dilation=dw_dilation[1],  # 膨胀率为2
        )
        # 第三个深度卷积：对第二分支（通道占比约4/8）应用7x7卷积
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1,
            dilation=dw_dilation[2],  # 膨胀率为3
        )
        # 逐点卷积，用于融合各分支特征
        self.PW_conv = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1
        )

    def forward(self, x):
        x_0 = self.DW_conv0(x)  # 对整个输入应用第一个5x5深度卷积

        # 从x_0中截取第二部分通道，输入到第二个深度卷积
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])

        # 从x_0中截取最后一部分通道，输入到第三个深度卷积
        x_2 = self.DW_conv2(x_0[:, self.embed_dims - self.embed_dims_2:, ...])

        # 在通道维度上拼接第一部分、第二部分和第三部分的输出
        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)

        x = self.PW_conv(x)  # 使用1x1卷积融合拼接后的多分支特征
        return x


class MultiOrderGatedAggregation(nn.Module):
    """
    实现多阶门控聚合的空间模块。
    
    参数:
        embed_dims (int): 输入特征图的通道数。
        attn_dw_dilation (list): 三个深度卷积层的膨胀因子。
        attn_channel_split (list): 各分支的通道分配比例。
        attn_force_fp32 (bool): 是否强制以FP32计算，默认为False。
    """
    
    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_force_fp32=False,
                 ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        # 第一个1x1卷积，用于初步特征投影
        self.proj_1 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        # 门控分支：生成门控系数
        self.gate = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        # 值分支：通过多阶深度卷积提取特征
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )

        # 最终融合的1x1卷积层
        self.proj_2 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # 使用SiLU激活函数分别激活门控和值分支
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

        # 使用ElementScale对特征进行微调分解
        self.sigma = ElementScale(embed_dims, init_value=1e-5, requires_grad=True)
        # ai缝合大王

    def feat_decompose(self, x):
        # 通过1x1卷积先进行特征投影
        x = self.proj_1(x)

        # 对投影后的特征进行全局平均池化，获得全局统计信息
        x_d = F.adaptive_avg_pool2d(x, output_size=1)

        # 利用sigma对原始特征与全局均值的差异进行微调，并将结果加回原始特征中
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)  # 应用激活函数，此处设计可根据需求调整
        return x

    def forward(self, x):
        shortcut = x.clone()  # 保存输入以便后续残差连接

        # 蓝色框部分：通过特征分解模块调整特征
        x = self.feat_decompose(x)

        # 灰色框部分：分别生成门控系数F和值特征G
        F_branch = self.gate(x)
        G_branch = self.value(x)

        # 分别对F和值特征应用SiLU激活后逐元素相乘，并通过proj_2融合
        x = self.proj_2(self.act_gate(F_branch) * self.act_gate(G_branch))
        x = x + shortcut  # 添加残差连接以保留原始信息
        # ai缝合大王
        return x

if __name__ == '__main__':

    input = torch.randn(1, 64, 32, 32)  # 生成一个随机输入张量，形状为(1, 64, 32, 32)

    MOGA = MultiOrderGatedAggregation(64)  # 实例化多阶门控聚合模块，设定通道数为64

    output = MOGA(input)
    print('MOGA_input_size:', input.size())
    print('MOGA_output_size:', output.size())
