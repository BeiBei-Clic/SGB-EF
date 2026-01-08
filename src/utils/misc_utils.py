import os
import glob
import torch
import argparse


def find_latest_checkpoint(args):
    """查找最新的检查点目录（Accelerate格式）"""
    save_dir = args.save_dir

    # 查找所有epoch检查点目录
    checkpoint_dirs = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*"))

    if checkpoint_dirs:
        # 提取epoch数字并返回最新的目录
        return max(checkpoint_dirs, key=lambda d: int(os.path.basename(d).split('_')[2]))

    # 查找final模型目录
    final_model_dir = os.path.join(save_dir, "continuous_flow_final")
    return final_model_dir if os.path.exists(final_model_dir) else None


def load_checkpoint(checkpoint_path, model, condition_encoder, device, optimizer=None, verbose=True, skip_config_json=False):
    """加载模型检查点（支持Accelerate格式）

    Args:
        checkpoint_path: 检查点路径
        model: 模型
        condition_encoder: 条件编码器
        device: 设备
        optimizer: 优化器（可选）
        verbose: 是否打印详细信息
        skip_config_json: 是否跳过加载training_config.json（推理模式下建议跳过，因为文件可能很大）
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None

    if verbose:
        print(f"Loading checkpoint: {checkpoint_path}")

    training_config = None

    # 如果不跳过配置文件，则加载
    if not skip_config_json:
        # 加载训练配置信息（如果存在）
        config_path = os.path.join(checkpoint_path, "training_config.json")
        if os.path.exists(config_path):
            # 添加所有可能的自定义类到安全全局列表
            torch.serialization.add_safe_globals([
                argparse.Namespace
            ])
            # 设置 weights_only=False 以支持加载包含自定义类的旧检查点
            training_config = torch.load(config_path, map_location=device, weights_only=False)

            def adjust_state_dict(state_dict, saved_was_dp, current_model):
                current_is_dp = hasattr(current_model, 'module')
                if saved_was_dp and not current_is_dp:
                    return {k.replace('module.', ''): v for k, v in state_dict.items()}
                elif not saved_was_dp and current_is_dp:
                    return {f'module.{k}': v for k, v in state_dict.items()}
                return state_dict

            # 加载模型权重（如果存在）
            if 'model_state_dict' in training_config:
                model_state = adjust_state_dict(
                    training_config['model_state_dict'],
                    training_config.get('model_was_dataparallel', False),
                    model
                )
                model.load_state_dict(model_state)
                if verbose:
                    print("✓ Model weights loaded from config")

            # 加载条件编码器权重（如果存在）
            if 'condition_encoder_state_dict' in training_config:
                encoder_state = adjust_state_dict(
                    training_config['condition_encoder_state_dict'],
                    training_config.get('encoder_was_dataparallel', False),
                    condition_encoder
                )
                condition_encoder.load_state_dict(encoder_state)
                if verbose:
                    print("✓ Condition encoder weights loaded from config")
        else:
            if verbose:
                print("  Note: training_config.json not found, model will be loaded by Accelerate")
    else:
        if verbose:
            print("  Skipping training_config.json (model weights will be loaded by Accelerate)")

    return training_config