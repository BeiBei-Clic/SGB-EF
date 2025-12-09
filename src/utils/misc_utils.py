import os
import glob


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