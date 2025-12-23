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

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
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

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    """Induced Set Attention Block"""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    """Pooling by Multihead Attention"""
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformerConditionEncoder(nn.Module):
    """使用SetTransformer架构编码残差点集为条件向量"""

    def __init__(self,
                 max_input_dim: int = 6,  # 支持的最大输入维度
                 dim_hidden: int = 128,
                 num_heads: int = 4,
                 num_inds: int = 32,
                 num_layers: int = 3,
                 num_seeds: int = 1,
                 dim_output: int = 128,
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
            print(f"  输出维度: {dim_output}")

        self.max_input_dim = max_input_dim
        self.dim_hidden = dim_hidden
        self.output_dim = dim_output

        # 预创建输入投影层，支持最大可能的输入维度 (max_input_dim x + 1 residual)
        max_feature_dim = max_input_dim + 1
        self.input_projection = nn.Linear(max_feature_dim, dim_hidden)

        # 添加权重初始化以避免梯度爆炸
        self._init_weights()

        # SetTransformer层
        self.first_layer = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        self.middle_layers = nn.ModuleList([
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
            for _ in range(num_layers - 1)
        ])

        # 聚合层
        self.pooling = PMA(dim_hidden, num_heads, num_seeds, ln=ln)

        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(dim_hidden * num_seeds, dim_hidden),  # 注意这里：输入是 dim_hidden * num_seeds
            nn.LayerNorm(dim_hidden),  # 添加LayerNorm防止梯度爆炸
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout
            nn.Linear(dim_hidden, dim_output)
        )

    def _init_weights(self):
        """初始化权重以避免梯度爆炸"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用较小的初始化
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_values: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        """
        使用SetTransformer编码残差点集 - 支持任意维度输入
        Args:
            x_values: (batch_size, num_points, input_dim) x值列表，input_dim可以是1,2,3,4...任意维度
            residuals: (batch_size, num_points) 残差值
        Returns:
            condition: (batch_size, output_dim) 条件向量
        """
        batch_size, num_points = residuals.shape
        input_dim = x_values.shape[2]  # 可以是1,2,3,4...任意维度

        # 检查输入维度是否在支持范围内
        if input_dim > self.max_input_dim:
            raise ValueError(f"输入维度 {input_dim} 超过了支持的最大维度 {self.max_input_dim}")

        # 构建输入特征: (batch_size, num_points, input_dim + 1)
        # 包含x的所有维度和residual值
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
        x = torch.nn.functional.gelu(x) * 0.1  # 缩小输出范围

        # 通过SetTransformer层
        x = self.first_layer(x)
        for layer in self.middle_layers:
            x = layer(x)

        # 聚合
        x = self.pooling(x)  # (batch_size, num_seeds, dim_hidden)

        # 展平并投影到输出维度
        x = x.view(batch_size, -1)  # (batch_size, num_seeds * dim_hidden)
        condition = self.output_projection(x)  # (batch_size, output_dim)

        # L2标准化（确保稳定的输出）
        condition = F.normalize(condition, p=2, dim=1, eps=1e-6)

        return condition


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