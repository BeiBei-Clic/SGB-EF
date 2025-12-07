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
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        cache_dir = os.path.join(project_root, "models", "huggingface_cache")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"加载模型: {model_name}")

        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side='left')
        self.output_dim = self.model.config.hidden_size

    def forward(self, x_values: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        """
        编码残差点集
        Args:
            x_values: (batch_size, num_points) x值
            residuals: (batch_size, num_points) 残差值
        Returns:
            condition: (batch_size, output_dim) 条件向量
        """
        # 确保输入为3维
        x_values = x_values.unsqueeze(-1) if x_values.dim() == 2 else x_values
        residuals = residuals.unsqueeze(-1) if residuals.dim() == 2 else residuals

        # 拼接并转换为文本
        points = torch.cat([x_values, residuals], dim=-1)
        task = "Generate numerical embeddings for symbolic regression data points"

        texts = []
        for b in range(points.shape[0]):
            point_texts = [f"Data point: x={points[b, i, 0].item():.6f}, r={points[b, i, 1].item():.6f}"
                          for i in range(points.shape[1])]
            texts.append(f"Instruct: {task}\nQuery: {' '.join(point_texts)}")

        # 编码文本
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(points.device)

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, -1]

        return torch.nn.functional.normalize(embeddings, p=2, dim=1)