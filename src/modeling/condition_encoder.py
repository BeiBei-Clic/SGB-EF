"""
条件编码器 - 使用SetTransformer架构编码残差点集
基于NeSymReS的SetTransformer实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    """Multi-Head Attention Block"""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, attention_mask):
        """
        Args:
            Q: (batch_size, n_q, dim_Q)
            K: (batch_size, n_k, dim_K)
            attention_mask: (batch_size, n_k) - 1表示有效位置，0表示填充位置
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # 计算attention scores
        A = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)

        # 应用attention mask
        batch_size, n_k = attention_mask.shape
        n_q = Q.size(1)

        # 扩展mask: (batch_size, n_k) -> (batch_size, 1, n_k) -> (batch_size, n_q, n_k)
        mask_expanded = attention_mask.unsqueeze(1).expand(-1, n_q, -1)

        # 重复每个头: (batch_size, n_q, n_k) -> (batch_size * num_heads, n_q, n_k)
        mask_expanded = mask_expanded.repeat(self.num_heads, 1, 1)

        # 将0位置设为极小值
        # 使用-1e4以兼容fp16（fp16范围约-65504到+65504）
        # 使用float类型确保在autocast下正确转换
        A = A.masked_fill(mask_expanded == 0, -1e4)

        A = torch.softmax(A, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    """Self-Attention Block"""
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, attention_mask):
        return self.mab(X, X, attention_mask)


class ISAB(nn.Module):
    """Induced Set Attention Block"""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, attention_mask):
        # mab0:诱导向量I作为Q，X作为K/V，需要对X应用mask
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, attention_mask)
        # mab1:X作为Q，H作为K/V，H是诱导向量不需要mask（全是有效）
        # 创建全1mask用于mab1（因为H没有padding）
        batch_size = X.size(0)
        num_inds = H.size(1)
        mask_for_H = torch.ones(batch_size, num_inds, device=X.device, dtype=X.dtype)
        return self.mab1(X, H, mask_for_H)


class PMA(nn.Module):
    """Pooling by Multihead Attention"""
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, attention_mask):
        # seed向量S作为Q，X作为K/V，需要对X应用mask
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, attention_mask)


class SetTransformerConditionEncoder(nn.Module):
    """使用SetTransformer架构编码残差点集为条件向量"""

    def __init__(self,
                 max_input_dim: int = 6,  # 支持的最大输入维度
                 dim_hidden: int = 128,
                 num_heads: int = 4,
                 num_inds: int = 32,
                 num_layers: int = 3,
                 num_seeds: int = 32,  # 修改：从 1 改为 32，输出多个特征向量
                 dim_output: int = 128,  # 保留参数以兼容旧代码，但不再使用
                 ln: bool = True,
                 input_normalization: bool = True,
                 verbose: bool = False):
        super().__init__()

        if verbose:
            print(f"初始化SetTransformer条件编码器:")
            print(f"  最大输入维度: {max_input_dim}")
            print(f"  隐藏层维度: {dim_hidden}")
            print(f"  注意力头数: {num_heads}")
            print(f"  诱导点数: {num_inds}")
            print(f"  层数: {num_layers}")
            print(f"  输出序列长度: {num_seeds}")
            print(f"  输出向量维度: {dim_hidden}")

        self.max_input_dim = max_input_dim
        self.dim_hidden = dim_hidden
        self.num_seeds = num_seeds  # 保存输出序列长度
        # self.output_dim = dim_output  # 不再需要，输出是序列而非单个向量

        # 输入特征维度：max_input_dim x + 1 residual
        encoded_feature_dim = max_input_dim + 1

        # 输入投影层
        self.input_projection = nn.Linear(encoded_feature_dim, dim_hidden)

        # 添加权重初始化以避免梯度爆炸
        self._init_weights()

        # SetTransformer层
        self.first_layer = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        self.middle_layers = nn.ModuleList([
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
            for _ in range(num_layers - 1)
        ])

        # 聚合层 - 输出序列而非单个向量
        self.pooling = PMA(dim_hidden, num_heads, num_seeds, ln=ln)

        # 不再需要 output_projection，因为我们直接输出序列
        # self.output_projection = nn.Sequential(
        #     nn.Linear(dim_hidden * num_seeds, dim_hidden),
        #     nn.LayerNorm(dim_hidden),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(dim_hidden, dim_output)
        # )

    def _init_weights(self):
        """初始化权重以避免梯度爆炸"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用较小的初始化
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_values: torch.Tensor, residuals: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        """
        使用SetTransformer编码残差点集 - 支持任意维度输入
        Args:
            x_values: (batch_size, num_points, input_dim) x值列表，input_dim可以是1,2,3,4...任意维度
            residuals: (batch_size, num_points) 残差值
            point_mask: (batch_size, num_points) 点掩码，1表示真实点，0表示填充点
        Returns:
            condition: (batch_size, num_seeds, dim_hidden) 条件序列，不再是单个向量
        """
        batch_size, num_points = residuals.shape
        input_dim = x_values.shape[2]  # 可以是1,2,3,4...任意维度

        # 检查输入维度是否在支持范围内
        if input_dim > self.max_input_dim:
            raise ValueError(f"输入维度 {input_dim} 超过了支持的最大维度 {self.max_input_dim}")

        # 构建输入特征: (batch_size, num_points, input_dim + 1)
        input_features = torch.cat([
            x_values,  # (batch_size, num_points, input_dim)
            residuals.unsqueeze(-1)  # (batch_size, num_points, 1)
        ], dim=-1)  # (batch_size, num_points, input_dim + 1)

        # 填充到最大维度以匹配预创建的投影层
        expected_dim = self.max_input_dim + 1
        actual_dim = input_features.shape[-1]
        if actual_dim < expected_dim:
            padding_size = expected_dim - actual_dim
            padding = torch.zeros(batch_size, num_points, padding_size,
                                 device=x_values.device, dtype=x_values.dtype)
            input_features = torch.cat([input_features, padding], dim=-1)

        # 输入嵌入
        x = self.input_projection(input_features)  # (batch_size, num_points, dim_hidden)
        # 使用温和的激活函数，避免极端输出
        x = torch.nn.functional.gelu(x) # 缩小输出范围

        # 通过SetTransformer层，传递point_mask
        x = self.first_layer(x, point_mask)
        for layer in self.middle_layers:
            x = layer(x, point_mask)

        # 聚合 - 输出序列而非单个向量
        x = self.pooling(x, point_mask)  # (batch_size, num_seeds, dim_hidden)

        # 不再展平和投影，直接返回序列
        # 这样每个 seed 都代表不同的特征簇，可以用于真正的交叉注意力
        # x = x.view(batch_size, -1)  # 删除这行
        # condition = self.output_projection(x)  # 删除这行

        return x  # (batch_size, num_seeds, dim_hidden)


# 保持向后兼容性，使用新的SetTransformer编码器
class ConditionEncoder(SetTransformerConditionEncoder):
    """条件编码器 - 现在使用SetTransformer架构"""

    def __init__(self,
                 model_name: str = None,  # 保持参数兼容性但不再使用
                 verbose: bool = False,
                 max_length: int = None,  # 保持参数兼容性但不再使用
                 **kwargs):
        args = kwargs['args']
        transformer_params = {
            'max_input_dim': getattr(args, 'condition_max_input_dim', 6),
            'dim_hidden': getattr(args, 'condition_dim_hidden', 128),
            'num_heads': getattr(args, 'condition_num_heads', 4),
            'num_inds': getattr(args, 'condition_num_inds', 32),
            'num_layers': getattr(args, 'condition_num_layers', 3),
            'num_seeds': getattr(args, 'condition_num_seeds', 1),
            'dim_output': getattr(args, 'condition_dim_output', 128),
            'verbose': verbose
        }

        super().__init__(**transformer_params)