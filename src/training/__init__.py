"""
训练模块

重构说明：
- EditFlowManager: 协调者，负责数据准备、模型设置和流程协调
- EditFlowTrainer: 训练器，专注于训练循环和评估
- InferenceEngine: 推理引擎，专注于符号回归推理
- FlowDataset, ContinuousFlowLoss: 数据集和损失函数
"""

from .editflow_manager import EditFlowManager
from .trainers.editflow_trainer import EditFlowTrainer
from .inference.inference_engine import InferenceEngine
from .flow import FlowDataset, ContinuousFlowLoss


__all__ = [
    'EditFlowManager',      # 主管理器（向后兼容）
    'EditFlowTrainer',      # 训练器
    'InferenceEngine',      # 推理引擎
    'FlowDataset',          # 数据集（向后兼容）
    'ContinuousFlowLoss',   # 损失函数（向后兼容）
]