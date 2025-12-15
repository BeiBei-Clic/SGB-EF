import os
import glob
import torch
import argparse
from ..modeling.editflow_transformer import EditFlowConfig


def find_latest_checkpoint(args):
    """查找最新的检查点文件"""
    save_dir = getattr(args, 'save_dir', 'checkpoints')

    # 查找所有epoch检查点
    checkpoint_files = glob.glob(os.path.join(save_dir, "editflow_epoch_*.pth"))

    if checkpoint_files:
        # 提取epoch数字并返回最新的
        return max(checkpoint_files, key=lambda f: int(os.path.basename(f).split('_')[2].split('.')[0]))

    # 查找final模型
    final_model = os.path.join(save_dir, "continuous_flow_final.pth")
    return final_model if os.path.exists(final_model) else None


def load_checkpoint(checkpoint_path, model, condition_encoder, device, optimizer=None, verbose=True):
    """加载模型检查点"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None

    if verbose:
        print(f"Loading checkpoint: {checkpoint_path}")

    torch.serialization.add_safe_globals([EditFlowConfig, argparse.Namespace])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    def adjust_state_dict(state_dict, saved_was_dp, current_model):
        current_is_dp = hasattr(current_model, 'module')
        if saved_was_dp and not current_is_dp:
            return {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not saved_was_dp and current_is_dp:
            return {f'module.{k}': v for k, v in state_dict.items()}
        return state_dict

    if 'model_state_dict' in checkpoint:
        model_state = adjust_state_dict(
            checkpoint['model_state_dict'],
            checkpoint.get('model_was_dataparallel', False),
            model
        )
        model.load_state_dict(model_state)
        if verbose:
            print("✓ Model loaded")

    if 'condition_encoder_state_dict' in checkpoint:
        encoder_state = adjust_state_dict(
            checkpoint['condition_encoder_state_dict'],
            checkpoint.get('encoder_was_dataparallel', False),
            condition_encoder
        )
        condition_encoder.load_state_dict(encoder_state)
        if verbose:
            print("✓ Condition encoder loaded")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if verbose:
            print("✓ Optimizer loaded")

    return checkpoint