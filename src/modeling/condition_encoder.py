"""
条件编码器 - 使用Hugging Face嵌入模型编码残差点集
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ConditionEncoder(nn.Module):
    """使用Hugging Face模型编码残差点集为条件向量"""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        super().__init__()

        # 设置本地缓存目录
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "huggingface_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # 加载Qwen模型和tokenizer到本地缓存
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side='left')

        self.output_dim = self.model.config.hidden_size

    def points_to_text(self, points: torch.Tensor) -> list[str]:
        batch_size = points.shape[0]
        texts = []

        for b in range(batch_size):
            point_texts = []
            for i in range(points.shape[1]):
                x_val = points[b, i, 0].item()
                r_val = points[b, i, 1].item()
                point_texts.append(f"Data point: x={x_val:.6f}, r={r_val:.6f}")

            text = " ".join(point_texts)
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

        # 添加任务指令
        task = "Generate numerical embeddings for symbolic regression data points"
        texts = [f"Instruct: {task}\nQuery: {text}" for text in texts]

        # 编码文本
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        device = points.device
        for key in inputs:
            inputs[key] = inputs[key].to(device)

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, -1]
        condition = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return condition


# 测试代码
if __name__ == "__main__":
    # 创建条件编码器
    encoder = ConditionEncoder(model_name="Qwen/Qwen3-Embedding-0.6B")

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