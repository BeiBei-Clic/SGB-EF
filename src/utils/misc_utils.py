import os
import glob
import torch
import argparse
from ..modeling.editflow_transformer import EditFlowConfig


def find_latest_checkpoint(args):
    """查找最新的检查点文件"""
    save_dir = getattr(args, 'save_dir', 'checkpoints')

    # 查找所有epoch检查点
    pattern = os.path.join(save_dir, "editflow_epoch_*.pth")
    checkpoint_files = glob.glob(pattern)

    if checkpoint_files:
        # 提取epoch数字并排序
        def get_epoch_number(filepath):
            filename = os.path.basename(filepath)
            epoch_str = filename.replace('editflow_epoch_', '').replace('.pth', '')
            return int(epoch_str)

        # 返回最新epoch的检查点
        latest_checkpoint = max(checkpoint_files, key=get_epoch_number)
        return latest_checkpoint

    # 如果没有epoch检查点，尝试final模型
    final_model = os.path.join(save_dir, "continuous_flow_final.pth")
    if os.path.exists(final_model):
        return final_model

    return None


def load_checkpoint(checkpoint_path, model, condition_encoder, device, optimizer=None):
    """
    从检查点加载模型状态
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 要加载状态的模型
        condition_encoder: 要加载状态的条件编码器
        device: 设备
        optimizer: 要加载状态的优化器（可选）
        
    Returns:
        checkpoint: 加载的检查点字典
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
        
    print(f"正在加载预训练模型: {checkpoint_path}")

    # 添加安全全局类以支持weights_only加载
    torch.serialization.add_safe_globals([
        EditFlowConfig,
        argparse.Namespace
    ])

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # 获取保存时的状态信息
    saved_model_was_dataparallel = checkpoint.get('model_was_dataparallel', False)
    saved_encoder_was_dataparallel = checkpoint.get('encoder_was_dataparallel', False)

    # 加载模型状态
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']

        # 根据保存和当前状态的差异，调整键名
        current_model_is_dataparallel = hasattr(model, 'module')

        if saved_model_was_dataparallel and not current_model_is_dataparallel:
            # 保存时是DataParallel，当前不是：移除module.前缀
            model_state = {key.replace('module.', ''): value for key, value in model_state.items()}
        elif not saved_model_was_dataparallel and current_model_is_dataparallel:
            # 保存时不是DataParallel，当前是：添加module.前缀
            model_state = {f'module.{key}': value for key, value in model_state.items()}

        model.load_state_dict(model_state)
        print("✓ EditFlow模型加载完成")

    # 加载条件编码器状态
    if 'condition_encoder_state_dict' in checkpoint:
        encoder_state = checkpoint['condition_encoder_state_dict']

        # 根据保存和当前状态的差异，调整键名
        current_encoder_is_dataparallel = hasattr(condition_encoder, 'module')

        if saved_encoder_was_dataparallel and not current_encoder_is_dataparallel:
            # 保存时是DataParallel，当前不是：移除module.前缀
            encoder_state = {key.replace('module.', ''): value for key, value in encoder_state.items()}
        elif not saved_encoder_was_dataparallel and current_encoder_is_dataparallel:
            # 保存时不是DataParallel，当前是：添加module.前缀
            encoder_state = {f'module.{key}': value for key, value in encoder_state.items()}

        condition_encoder.load_state_dict(encoder_state)
        print("✓ 条件编码器加载完成")

    # 如果有优化器，加载优化器状态
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ 优化器状态加载完成")

    return checkpoint