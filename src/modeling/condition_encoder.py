"""
条件编码器 - 使用Hugging Face嵌入模型编码残差点集
"""

import os
import warnings
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# 在模块级别过滤transformers的词汇扩展警告
warnings.filterwarnings("ignore", message=".*mean_resizing.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*multivariate normal distribution.*", category=UserWarning)



class ConditionEncoder(nn.Module):
    """使用Hugging Face模型编码残差点集为条件向量"""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        super().__init__()

        # 设置本地缓存目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        cache_dir = os.path.join(project_root, "models", "huggingface_cache")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"加载模型: {model_name}")

        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side='left', trust_remote_code=True)
        self.output_dim = self.model.config.hidden_size

    def forward(self, x_values: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        """
        编码残差点集 - 支持任意维度输入
        Args:
            x_values: (batch_size, num_points, input_dim) x值列表，input_dim可以是1,2,3,4...任意维度
            residuals: (batch_size, num_points) 残差值
        Returns:
            condition: (batch_size, output_dim) 条件向量
        """
        batch_size, num_points = residuals.shape
        input_dim = x_values.shape[2]  # 可以是1,2,3,4...任意维度

        # 将数据点转换为文本
        task = "Generate numerical embeddings for symbolic regression data points"
        texts = []

        for b in range(batch_size):
            point_texts = []
            for i in range(num_points):
                # 统一处理：将x的所有维度作为一个列表，支持任意维度(1D,2D,3D,4D,...)
                x_vals = [x_values[b, i, d].item() for d in range(input_dim)]
                x_str = ','.join(f"{x_val:.6f}" for x_val in x_vals)
                r_val = residuals[b, i].item()
                point_texts.append(f"x: [{x_str}], r={r_val:.6f}")

            texts.append(f"Instruct: {task}\nQuery: {' '.join(point_texts)}")
        # print(texts)
        # 编码文本
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(x_values.device)

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, -1]

        return torch.nn.functional.normalize(embeddings, p=2, dim=1)