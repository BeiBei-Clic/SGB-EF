"""
条件编码器 - 使用Hugging Face嵌入模型编码残差点集
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ConditionEncoder(nn.Module):
    """使用Hugging Face模型编码残差点集为条件向量"""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()

        # 加载预训练模型和tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 添加padding token如果不存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.output_dim = self.model.config.hidden_size

    def points_to_text(self, points: torch.Tensor) -> list[str]:
        """
        将点集直接转换为文本序列
        Args:
            points: (batch_size, num_points, 2) 点集数据 (x, r)
        Returns:
            texts: 文本序列列表
        """
        batch_size = points.shape[0]
        texts = []

        for b in range(batch_size):
            # 直接将点对转换为文本
            point_pairs = []
            for i in range(points.shape[1]):
                x_val = points[b, i, 0].item()
                r_val = points[b, i, 1].item()
                point_pairs.append(f"{x_val:.6f},{r_val:.6f}")

            # 用空格分隔所有点对
            text = " ".join(point_pairs)
            texts.append(text)

        return texts

    def forward(self, x_values: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        """
        编码残差点集
        Args:
            x_values: (batch_size, num_points) x值
            residuals: (batch_size, num_points) 残差值
        Returns:
            condition: (batch_size, output_dim) 条件向量
        """
        # 确保输入形状正确
        if x_values.dim() == 2:
            x_values = x_values.unsqueeze(-1)  # (batch_size, num_points, 1)

        # 拼接x值和残差
        points = torch.cat([x_values, residuals.unsqueeze(-1)], dim=-1)
        # (batch_size, num_points, 2)

        # 转换为文本
        texts = self.points_to_text(points)

        # 编码文本
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # 移动到正确设备
        device = points.device
        for key in inputs:
            inputs[key] = inputs[key].to(device)

        # 通过模型获取嵌入
        outputs = self.model(**inputs)

        # 使用pooled output或最后一个hidden state的平均值
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            condition = outputs.pooler_output
        else:
            condition = outputs.last_hidden_state.mean(dim=1)

        return condition


# 测试代码
if __name__ == "__main__":
    # 创建条件编码器
    encoder = ConditionEncoder(model_name="distilbert-base-uncased")

    # 测试数据
    batch_size = 2
    num_points = 10
    x_values = torch.randn(batch_size, num_points)
    residuals = torch.randn(batch_size, num_points)

    # 测试编码
    with torch.no_grad():
        condition = encoder(x_values, residuals)
        print(f"条件编码输出形状: {condition.shape}")
        print(f"条件编码输出均值: {condition.mean().item():.4f}")

    print("Hugging Face条件编码器测试完成！")