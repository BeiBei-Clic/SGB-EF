"""
条件编码器 - 使用SetTransformer架构编码残差点集
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        Q_, K_, V_ = torch.cat(Q.split(dim_split, 2), 0), torch.cat(K.split(dim_split, 2), 0), torch.cat(V.split(dim_split, 2), 0)

        # 计算注意力并应用mask
        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        A = A.masked_fill(
            attention_mask.unsqueeze(1).expand(-1, Q.size(1), -1).repeat(self.num_heads, 1, 1) == 0,
            -1e4
        )
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
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, attention_mask)
        # H是诱导向量，没有padding，创建全1mask
        return self.mab1(X, H, torch.ones(X.size(0), H.size(1), device=X.device, dtype=X.dtype))


class PMA(nn.Module):
    """Pooling by Multihead Attention"""
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, attention_mask):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, attention_mask)


class SetTransformerConditionEncoder(nn.Module):
    """使用SetTransformer架构编码残差点集为条件向量"""

    def __init__(self,
                 max_input_dim: int = 3,
                 dim_hidden: int = 128,
                 num_heads: int = 4,
                 num_inds: int = 32,
                 num_layers: int = 3,
                 num_seeds: int = 32,
                 dim_output: int = 128,
                 ln: bool = True,
                 input_normalization: bool = True,
                 verbose: bool = False):
        super().__init__()

        if verbose:
            print(f"初始化SetTransformer条件编码器: max_input_dim={max_input_dim}, dim_hidden={dim_hidden}, num_heads={num_heads}, num_inds={num_inds}, num_layers={num_layers}, num_seeds={num_seeds}")

        self.max_input_dim = max_input_dim
        self.dim_hidden = dim_hidden

        # 输入投影层
        self.input_projection = nn.Linear(max_input_dim + 1, dim_hidden)
        self._init_weights()

        # SetTransformer层
        self.first_layer = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        self.middle_layers = nn.ModuleList([
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
            for _ in range(num_layers - 1)
        ])

        # 聚合层
        self.pooling = PMA(dim_hidden, num_heads, num_seeds, ln=ln)

    def _init_weights(self):
        """初始化权重以避免梯度爆炸"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_values: torch.Tensor, y_target: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        """
        使用SetTransformer编码目标值点集

        Args:
            x_values: (batch_size, num_points, input_dim) x值
            y_target: (batch_size, num_points) 目标值
            point_mask: (batch_size, num_points) 点掩码，1表示真实点，0表示填充

        Returns:
            condition: (batch_size, num_seeds, dim_hidden) 条件序列
        """
        batch_size, num_points = y_target.shape
        input_dim = x_values.shape[2]

        if input_dim > self.max_input_dim:
            raise ValueError(f"输入维度 {input_dim} 超过了支持的最大维度 {self.max_input_dim}")

        # 构建输入特征：[x0, x1, ..., xN, 0, ..., 0, y_target]
        input_features = torch.zeros(batch_size, num_points, self.max_input_dim + 1, device=x_values.device, dtype=x_values.dtype)
        input_features[:, :, :input_dim] = x_values
        input_features[:, :, -1] = y_target

        # 输入嵌入
        x = F.gelu(self.input_projection(input_features))

        # 通过SetTransformer层
        x = self.first_layer(x, point_mask)
        for layer in self.middle_layers:
            x = layer(x, point_mask)

        # 聚合
        return self.pooling(x, point_mask)

