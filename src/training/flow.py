"""
EditFlow连续流匹配的核心组件
"""

import torch
import json
import os
import time
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from ..symbolic.data_generator import generate_flow_samples


def remove_gap_tokens(z_t: torch.Tensor, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """移除gap token并返回处理后的序列

    重要：此函数只移除 gap_token，保留 BOS token 和所有其他 tokens
    确保返回的序列格式为 [BOS] + [non_gap_tokens] + [PAD...]
    """
    pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
    gap_token_id = tokenizer.convert_tokens_to_ids('<gap>')
    bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
    batch_size, z_seq_len = z_t.shape
    device = z_t.device

    z_gap_mask = (z_t == gap_token_id)
    z_pad_mask = (z_t == pad_token_id)

    # 使用掩码操作移除gap tokens（只移除gap，保留BOS和其他tokens）
    x_t_list = []
    for i in range(batch_size):
        non_gap_mask = ~z_gap_mask[i]
        x_row = z_t[i][non_gap_mask]
        x_t_list.append(x_row)

        # 验证：确保BOS token被保留（如果输入中存在）
        if len(x_row) > 0 and z_t[i, 0] == bos_token_id:
            assert x_row[0] == bos_token_id, f"BOS token必须被保留在位置0"

    max_len = max(len(x) for x in x_t_list)
    x_t_padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    x_pad_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for i, x_row in enumerate(x_t_list):
        x_t_padded[i, :len(x_row)] = x_row
        x_pad_mask_padded[i, :len(x_row)] = (x_row == pad_token_id)

    return x_t_padded, x_pad_mask_padded, z_gap_mask, z_pad_mask


def fill_gap_tokens_with_repeats(x_ut: torch.Tensor, z_gap_mask: torch.Tensor, z_pad_mask: torch.Tensor) -> torch.Tensor:
    """用重复值填充gap token位置"""
    batch_size, z_seq_len = z_gap_mask.shape
    _, x_seq_len, vocab_size = x_ut.shape


    # 计算每个位置对应的非gap位置

    non_gap_mask = ~z_gap_mask


    indices = non_gap_mask.cumsum(dim=1) - 1


    indices = indices.clamp(min=0, max=x_seq_len-1)


    # 收集对应的特征
    batch_indices = torch.arange(batch_size, device=x_ut.device).unsqueeze(1)

    result = x_ut[batch_indices, indices]

    result[z_pad_mask] = 0


    return result


class ContinuousFlowLoss:
    """连续时间流匹配损失函数（架构v2.0 - 固定t=0，不再需要调度器）"""

    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

    def make_ut_mask_from_z(self, z_t: torch.Tensor, z_1: torch.Tensor, vocab_size: int,
                           gap_token: int, tokenizer, x_t: torch.Tensor) -> torch.Tensor:
        """
        根据论文Fig. 13的双索引追踪逻辑，生成正确的编辑操作掩码

        核心思想：在Z空间（含gap）遍历，动态维护X空间（无gap）的索引指针

        Args:
            z_t: 当前状态（Z空间，含gap）[batch, z_seq_len]
            z_1: 目标状态（Z空间，含gap）[batch, z_seq_len]
            vocab_size: 词汇表大小
            gap_token: gap token的ID
            tokenizer: 分词器
            x_t: 当前状态（X空间，无gap）[batch, x_seq_len] - 用于双索引映射

        Returns:
            u_mask: 编辑操作掩码 [batch, x_seq_len, 2*vocab_size+1]
                    使用one-hot编码：u_mask[b, pos, op_id] = 1 表示在位置pos执行操作op_id
                    每个位置对应：[vocab_size个插入操作, vocab_size个替换操作, 1个删除操作]
        """
        batch_size, z_seq_len = z_t.shape
        x_seq_len = x_t.shape[1]
        n_ops = 2 * vocab_size + 1  # 插入(vocab_size) + 替换(vocab_size) + 删除(1)

        pad_token = tokenizer.convert_tokens_to_ids('<pad>')

        # 初始化输出掩码（在X空间）
        # ✅ 使用one-hot编码：u_mask[b, pos, op_id] = 1 表示在位置pos执行操作op_id
        # 即使需要多次插入同一token，也只标记一次（因为这是操作类型，不是数量）
        u_mask = torch.zeros((batch_size, x_seq_len, n_ops), dtype=torch.int, device=z_t.device)

        # 对每个样本进行双索引遍历（论文Fig. 13的核心逻辑）
        for b in range(batch_size):
            x_t_index = -1  # X空间指针初始化为-1（指向x_t的前一个位置）
            first_valid_index = 0  # 记录第一个有效token的位置，用于处理开头的gap

            for i in range(z_seq_len):
                token_t = z_t[b, i].item()
                token_1 = z_1[b, i].item()

                # 跳过z_t和z_1的pad位置
                if token_t == pad_token or token_1 == pad_token:
                    continue

                # === 关键步骤1：维护X空间指针 ===
                # 如果z_t当前位置不是gap，说明它在x_t中占据一个位置
                # 因此需要将x_t_index向前移动一位
                if token_t != gap_token:
                    x_t_index += 1  # 移动到x_t中的下一个位置

                    # 更新第一个有效token的位置
                    if first_valid_index == 0:
                        first_valid_index = x_t_index

                    # === 关键修复：检查x_t当前位置是否是pad ===
                    # 如果x_t当前位置是pad，说明已经超出有效长度，停止遍历
                    if x_t_index >= x_seq_len:
                        break  # 超出x_t的有效长度，停止

                    if x_t[b, x_t_index].item() == pad_token:
                        break  # x_t当前位置是pad，说明已经是填充区域，停止

                # === 关键步骤2：判断编辑类型并标记 ===
                # 根据z_t[i]和z_1[i]的关系，决定在x_t[x_t_index]位置执行什么操作

                if token_t == gap_token and token_1 != gap_token:
                    # 插入操作：
                    # z_t[i]是gap，z_1[i]是有效token
                    # 意味着需要在gap位置插入token_1
                    #
                    # ✅ 修复：使用one-hot编码而非累加计数
                    # 即使需要多次插入相同token，也只标记一次
                    # 因为模型只需要知道"在这个位置执行插入sin操作"
                    # 而不是"插入2次sin"（后者是编辑操作的定义，不是损失的一部分）
                    insert_pos = max(x_t_index, first_valid_index)
                    if insert_pos >= 0 and insert_pos < x_seq_len:
                        u_mask[b, insert_pos, token_1] = 1  # one-hot编码（只标记操作类型）

                elif token_t != gap_token and token_1 == gap_token:
                    # 删除操作：
                    # z_t[i]是有效token，z_1[i]是gap
                    # 意味着需要删除当前token
                    # 删除操作直接标记在x_t_index位置（当前token的位置）
                    if x_t_index >= 0 and x_t_index < x_seq_len:
                        u_mask[b, x_t_index, -1] = 1  # one-hot编码

                elif token_t != gap_token and token_1 != gap_token and token_t != token_1:
                    # 替换操作：
                    # z_t[i]和z_1[i]都是有效token但不同
                    # 意味着需要将token_t替换为token_1
                    # 替换操作标记在x_t_index位置（偏移vocab_size以区分插入和替换）
                    if x_t_index >= 0 and x_t_index < x_seq_len:
                        u_mask[b, x_t_index, token_1 + vocab_size] = 1  # one-hot编码

        return u_mask

    def __call__(self, u_cat_x: torch.Tensor, u_z: torch.Tensor, u_mask: torch.Tensor,
                 vocab_size: int, accelerator=None, logger=None) -> torch.Tensor:
        """
        连续流损失计算

        Args:
            u_cat_x: X空间的预测速率 [batch, x_seq_len, 2*vocab_size+1]（原始空间，不含gap）
            u_z: Z空间的预测速率 [batch, z_seq_len, 2*vocab_size+1]（扩展空间，含gap重复）
            u_mask: 操作掩码 [batch, z_seq_len, 2*vocab_size+1]
            vocab_size: 词汇表大小
            accelerator: Accelerate加速器
            logger: 日志记录器（可选）

        Returns:
            loss: 标量损失值
        """
        # NaN/Inf检测 - 检测两个输入
        nan_count_x = torch.isnan(u_cat_x).sum().item()
        inf_count_x = torch.isinf(u_cat_x).sum().item()
        nan_count_z = torch.isnan(u_z).sum().item()
        inf_count_z = torch.isinf(u_z).sum().item()

        # 额外检查：u_z 是否包含接近0的值（可能导致 log(0)）
        u_z_min = float(u_z.min().item())
        u_z_max = float(u_z.max().item())
        u_z_mean = float(u_z.mean().item())
        u_z_std = float(u_z.std().item())
        u_z_has_near_zero = bool((u_z < 1e-10).any().item())
        u_z_has_negative = bool((u_z < 0).any().item())
        u_z_num_zeros = int((u_z == 0).sum().item())

        # 分布式NaN检测：如果使用accelerator，确保所有进程同步检测结果
        has_nan_or_inf = torch.tensor(1 if ((nan_count_x > 0 or inf_count_x > 0) or (nan_count_z > 0 or inf_count_z > 0)) else 0,
                                    device=u_cat_x.device)

        if accelerator is not None and accelerator.distributed_type != "NO":
            # 同步所有进程的NaN检测结果
            # 使用accelerator.gather收集所有进程的NaN检测结果
            gathered_results = accelerator.gather(has_nan_or_inf)
            has_nan_or_inf = gathered_results.sum()
        else:
            has_nan_or_inf = has_nan_or_inf.item()

        # 记录NaN/Inf检测结果（仅用于监控，不再抛出异常跳过）
        if has_nan_or_inf > 0 and logger is not None and accelerator.is_local_main_process:
            logger.error("INPUT_NAN_INF",
                        f"检测到异常值: X空间NaN={nan_count_x}, Inf={inf_count_x}, "
                        f"Z空间NaN={nan_count_z}, Inf={inf_count_z}, 分布式检测={has_nan_or_inf}",
                        "compute_loss", level=1)

        # 关键修复：u_total 需要在概率空间计算
        # 从logits转换为概率
        u_cat_x_probs = torch.exp(u_cat_x.clamp(max=10))  # 防止溢出
        u_total = u_cat_x_probs.sum(dim=(1, 2))

        # 🔧 关键修复：避免-inf污染loss计算
        # 问题：如果直接对所有位置计算log_softmax，padding位置的-inf会影响数值稳定性
        # 解决：对每个样本，只提取有效位置（u_mask=1）的logits，单独计算log_softmax

        batch_size, z_seq_len, n_ops = u_z.shape

        # 初始化输出张量，填充一个较小的负数（不是-inf，避免污染）
        log_u_z = torch.zeros_like(u_z) - 10.0  # 初始化为log(很小的概率)

        # 对每个样本单独处理
        for b in range(batch_size):
            # 找到该样本所有有效位置的mask
            mask_b = u_mask[b].bool()  # [z_seq_len, n_ops]

            if mask_b.any():
                # 提取有效位置的logits
                valid_logits = u_z[b][mask_b]  # [num_valid]

                # 只在有效位置计算log_softmax
                valid_log_probs = torch.log_softmax(valid_logits, dim=-1)

                # 填充回原位置
                log_u_z[b][mask_b] = valid_log_probs
            # 如果没有有效位置，保持初始值-10.0（log(exp(-10)) = -10）

        # 统计信息：log_u_z（应该在负无穷到0之间）
        log_u_z_min = float(log_u_z.min().item())
        log_u_z_max = float(log_u_z.max().item())
        log_u_z_mean = float(log_u_z.mean().item())
        log_u_z_has_inf = bool(torch.isinf(log_u_z).any().item())
        log_u_z_has_nan = bool(torch.isnan(log_u_z).any().item())

        # 统计信息：u_mask（标记需要预测的位置）
        u_mask_num_true = int(u_mask.sum().item())
        u_mask_total = int(u_mask.numel())
        u_mask_sparsity = float(1.0 - (u_mask_num_true / u_mask_total))

        # cross_entropy 在 Z 空间计算（负对数似然）
        # u_mask 标记了正确的操作位置（one-hot编码）
        # 只在 u_mask=True 的位置累加（其他位置不影响损失）
        masked_log_u_z = log_u_z * u_mask.float()

        # 🔧 修复：计算每个样本的有效操作数量（用于归一化）
        # 这确保不同长度的序列对损失的贡献是公平的
        valid_ops_per_sample = u_mask.sum(dim=(1, 2)).clamp(min=1)  # 至少为1避免除0

        # 计算每个样本的平均负对数似然（对序列长度归一化）
        cross_entropy = masked_log_u_z.sum(dim=(1, 2)) / valid_ops_per_sample

        # 统计信息：cross_entropy（负对数似然，应该是负数）
        cross_entropy_min = float(cross_entropy.min().item())
        cross_entropy_max = float(cross_entropy.max().item())
        cross_entropy_mean = float(cross_entropy.mean().item())
        cross_entropy_std = float(cross_entropy.std().item() if cross_entropy.numel() > 1 else 0.0)

        # 最终损失：负对数似然 + u_total正则化
        # cross_entropy: 惩罚"在需要编辑的位置预测错误"
        # u_total: 惩罚"预测过高的总操作速率"（鼓励稀疏性）
        # 这样模型同时学习两个目标：
        # 1. 在需要编辑的位置预测正确的编辑操作
        # 2. 在所有位置保持低操作速率（稀疏预测）

        # u_total已经在前面计算：u_total = u_cat_x.sum(dim=(1, 2))
        # shape: [batch]，每个样本的总操作速率

        # 🔧 测试：完全移除正则化，看能否过拟合
        reg_coeff = 0.0  # 移除所有正则化

        # 最终损失 = 负对数似然 + u_total正则化
        ce_loss = -cross_entropy.mean()
        u_total_loss = u_total.mean()
        loss = ce_loss + reg_coeff * u_total_loss

        # 统计信息：loss
        loss_value = float(loss.item())
        loss_is_nan = bool(torch.isnan(loss).item())
        loss_is_inf = bool(torch.isinf(loss).item())

        # 额外统计：分解损失项
        ce_loss_value = float(ce_loss.item())
        u_total_loss_value = float(u_total_loss.item())

        # 记录所有统计信息到日志（仅在debug模式下）
        if logger is not None and self.debug_mode:
            # 添加归一化相关的统计信息
            valid_ops_min = int(valid_ops_per_sample.min().item())
            valid_ops_max = int(valid_ops_per_sample.max().item())
            valid_ops_mean = float(valid_ops_per_sample.float().mean().item())

            logger.log(f"LOSS_STATS",
                      f"u_z: min={u_z_min:.6f}, max={u_z_max:.6f}, mean={u_z_mean:.6f}, std={u_z_std:.6f} | "
                      f"zeros={u_z_num_zeros}, near_zero={u_z_has_near_zero}, negative={u_z_has_negative} | "
                      f"log_u_z: min={log_u_z_min:.6f}, max={log_u_z_max:.6f}, mean={log_u_z_mean:.6f} | "
                      f"has_inf={log_u_z_has_inf}, has_nan={log_u_z_has_nan} | "
                      f"u_mask: {u_mask_num_true}/{u_mask_total} ({(1-u_mask_sparsity)*100:.2f}%) | "
                      f"valid_ops_per_sample: min={valid_ops_min}, max={valid_ops_max}, mean={valid_ops_mean:.2f} | "
                      f"cross_entropy: min={cross_entropy_min:.6f}, max={cross_entropy_max:.6f}, "
                      f"mean={cross_entropy_mean:.6f}, std={cross_entropy_std:.6f} | "
                      f"loss: {loss_value:.6f}, is_nan={loss_is_nan}, is_inf={loss_is_inf}",
                      level=2)

        return loss


def prepare_dataset_hf(data_file: str, tokenizer, max_dim: int = 10,
                       max_expr_length: int = 128, stream: bool = True,
                       num_proc: Optional[int] = None):
    """
    使用 Hugging Face datasets 加载并预处理数据

    Args:
        data_file: 数据文件路径 (.txt格式，每行一个JSON样本)
        tokenizer: 分词器
        max_dim: 最大维度
        max_expr_length: 表达式最大长度
        stream: 是否使用流式加载（默认True，适合大文件）
        num_proc: 预处理时的进程数，None表示自动选择

    Returns:
        dataset: Hugging Face Dataset 对象
    """
    data_files = {"train": data_file}

    # 加载原始文本数据
    if stream:
        # 流式加载：适合大文件，不一次性加载到内存
        raw_dataset = load_dataset("text", data_files=data_files, split="train", streaming=True)
    else:
        # 一次性加载：适合小文件，后续处理更快
        raw_dataset = load_dataset("text", data_files=data_files, split="train")

    # 获取token相关信息
    pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
    bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
    gap_token_id = tokenizer.convert_tokens_to_ids('<gap>')
    vocab_size = len(tokenizer.get_vocab())

    def process_function(examples):
        """
        预处理函数：将JSON字符串转换为模型所需的格式
        """
        # 处理单个样本（streaming模式，batch_size=1时）
        if isinstance(examples, dict) and 'text' in examples:
            text = examples['text']
            # 检查是否已经是list（streaming模式可能批处理）
            if isinstance(text, list):
                lines = text
            else:
                lines = [text]
        else:
            # 处理batch（非streaming模式）
            lines = examples['text']

        batch_size = len(lines)

        # 预分配列表
        outputs = {
            'x_values': [],
            'y_target': [],
            'residuals': [],
            'z0_token_ids': [],
            'z1_token_ids': [],
            'gap_token': []
        }

        for line in lines:
            sample = json.loads(line)

            # 添加数值数据
            outputs['x_values'].append(sample['x_values'])
            outputs['y_target'].append(sample['y_target'])
            outputs['residuals'].append(sample['residuals'])

            # Token处理
            def pad_z_sequence(tokens):
                # 过滤掉None值，并确保所有token都是整数
                filtered_tokens = [t for t in tokens if t is not None and isinstance(t, int)]
                if len(filtered_tokens) != len(tokens):
                    print(f"警告: 过滤了 {len(tokens) - len(filtered_tokens)} 个无效token")

                # 添加BOS token并截断
                tokens = [bos_token_id] + filtered_tokens[:max_expr_length-1]
                # Padding到固定长度
                tokens.extend([pad_token_id] * (max_expr_length - len(tokens)))
                return tokens

            # 转换token
            z0_tokens = tokenizer.convert_tokens_to_ids(sample['z0_tokens'])
            z1_tokens = tokenizer.convert_tokens_to_ids(sample['z1_tokens'])

            outputs['z0_token_ids'].append(pad_z_sequence(z0_tokens))
            outputs['z1_token_ids'].append(pad_z_sequence(z1_tokens))
            outputs['gap_token'].append(gap_token_id)

        return outputs

    # 应用预处理
    if stream:
        # 流式模式：使用map (IterableDataset不支持desc参数)
        tokenized_dataset = raw_dataset.map(
            process_function,
            batched=True,
            remove_columns=["text"]
        )
    else:
        # 非流式模式：使用多进程加速
        tokenized_dataset = raw_dataset.map(
            process_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=["text"],
            desc="Preprocessing dataset"
        )

    # 设置格式为torch，这样DataLoader拿到的直接是Tensor
    if stream:
        # IterableDataset使用with_format (不支持columns参数)
        tokenized_dataset = tokenized_dataset.with_format(type='torch')
    else:
        # 普通Dataset使用set_format
        tokenized_dataset.set_format(type='torch', columns=[
            'x_values', 'y_target', 'residuals',
            'z0_token_ids', 'z1_token_ids', 'gap_token'
        ])

    # 保存tokenizer引用供后续使用
    tokenized_dataset.tokenizer = tokenizer

    return tokenized_dataset


class FlowDataset(torch.utils.data.Dataset):
    """连续流数据集包装器 - 兼容旧接口，内部使用 Hugging Face datasets"""

    def __init__(self, data_file: str, tokenizer, max_dim: int = 10,
                 max_expr_length: int = 128, stream: bool = True,
                 num_proc: Optional[int] = None):
        """
        使用 Hugging Face datasets 的数据集包装器

        Args:
            data_file: 数据文件路径
            tokenizer: 分词器
            max_dim: 最大维度
            max_expr_length: 表达式最大长度
            stream: 是否使用流式加载（默认True）
            num_proc: 预处理时的进程数
        """
        self.tokenizer = tokenizer
        self.max_dim = max_dim
        self.max_expr_length = max_expr_length
        self.vocab_size = len(tokenizer.get_vocab())
        self.pad_token = tokenizer.convert_tokens_to_ids('<pad>')
        self.bos_token = tokenizer.convert_tokens_to_ids('<s>')
        self.gap_token = tokenizer.convert_tokens_to_ids('<gap>')
        self.stream = stream  # 保存stream模式标志

        # 使用 Hugging Face datasets 加载数据
        self._hf_dataset = prepare_dataset_hf(
            data_file=data_file,
            tokenizer=tokenizer,
            max_dim=max_dim,
            max_expr_length=max_expr_length,
            stream=stream,
            num_proc=num_proc
        )

        # 如果是非流式模式，缓存数据列表以便快速访问
        if not stream:
            self._data_list = list(self._hf_dataset)
            self._dataset_length = len(self._data_list)
        else:
            self._data_list = None
            # 流式模式：直接统计文件行数（比遍历dataset快得多）
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    self._dataset_length = sum(1 for _ in f)
            except Exception as e:
                # 如果文件统计失败，使用一个默认的大数
                print(f"警告: 无法统计文件行数，使用默认值。错误: {e}")
                self._dataset_length = 1000000  # 默认值

    def __len__(self):
        """返回数据集大小"""
        return self._dataset_length

    def __iter__(self):
        """流式模式下的迭代器"""
        if self._data_list is not None:
            # 非流式模式：迭代缓存列表
            return iter(self._data_list)
        else:
            # 流式模式：直接迭代Hugging Face dataset
            return self._hf_dataset.__iter__()

    def __getitem__(self, idx):
        """获取单个样本（仅非流式模式）"""
        if self._data_list is not None:
            # 非流式模式：直接从缓存列表获取
            sample = self._data_list[idx]
        else:
            # 流式模式：使用islice跳转到指定位置
            # 注意：在DataLoader中使用IterableDataset时，__getitem__不应该被调用
            from itertools import islice
            sample = next(islice(self._hf_dataset, idx, None))

        # 转换为Tensor（如果是numpy的话）
        result = {
            'x_values': torch.FloatTensor(sample['x_values'])
                if not isinstance(sample['x_values'], torch.Tensor) else sample['x_values'],
            'y_target': torch.FloatTensor(sample['y_target'])
                if not isinstance(sample['y_target'], torch.Tensor) else sample['y_target'],
            'residuals': torch.FloatTensor(sample['residuals'])
                if not isinstance(sample['residuals'], torch.Tensor) else sample['residuals'],
            'z0_token_ids': sample['z0_token_ids'].long()
                if not isinstance(sample['z0_token_ids'], torch.Tensor) else sample['z0_token_ids'],
            'z1_token_ids': sample['z1_token_ids'].long()
                if not isinstance(sample['z1_token_ids'], torch.Tensor) else sample['z1_token_ids'],
            'gap_token': sample['gap_token'].item()
                if isinstance(sample['gap_token'], torch.Tensor) else sample['gap_token']
        }

        return result


def custom_collate_fn(batch):
    """处理不同维度数据的collate函数，使用padding + mask方案

    修改：添加y_target的处理（架构改进：使用目标值而非残差作为条件）
    """
    if len(batch) == 0:
        return {
            'x_values': torch.empty(0, 0, 0),
            'y_target': torch.empty(0, 0),  # 修改：添加y_target
            'residuals': torch.empty(0, 0),
            'dim_mask': torch.empty(0, 0),
            'z0_token_ids': torch.empty(0, 0),
            'z1_token_ids': torch.empty(0, 0),
            'gap_token': None
        }

    # 找到最大维度和n_points
    max_dim = 0
    max_n_points = 0
    original_dims = []

    for i, item in enumerate(batch):
        x_val = item['x_values']  # [n_points, current_dim]
        y_tgt = item['y_target']  # [n_points]  # 新增：获取y_target

        # 确保x_values至少是2维的
        if x_val.dim() == 1:
            x_val = x_val.unsqueeze(1)  # [n_points, 1]

        # 确保y_target是1维的
        if y_tgt.dim() > 1:
            y_tgt = y_tgt.squeeze()

        current_dim = x_val.shape[1]
        current_n_points = x_val.shape[0]

        max_dim = max(max_dim, current_dim)
        max_n_points = max(max_n_points, current_n_points)
        original_dims.append(current_dim)

    # Padding所有数据到最大形状
    x_values_padded = []
    y_target_padded = []  # 新增：y_target的padding
    residuals_padded = []
    dim_masks = []
    point_masks = []

    for i, item in enumerate(batch):
        x_val = item['x_values'].clone()  # [n_points, current_dim]
        y_tgt = item['y_target'].clone()  # [n_points]  # 新增：克隆y_target
        resid = item['residuals'].clone()  # [n_points]  # 保留：用于向后兼容

        # 确保x_values至少是2维的
        if x_val.dim() == 1:
            x_val = x_val.unsqueeze(1)

        # 确保y_target和residuals是1维的
        if y_tgt.dim() > 1:
            y_tgt = y_tgt.squeeze()
        if resid.dim() > 1:
            resid = resid.squeeze()

        current_n_points = x_val.shape[0]
        current_dim = x_val.shape[1]

        # Padding n_points维度（如果需要）
        if current_n_points < max_n_points:
            padding_points = torch.zeros(max_n_points - current_n_points, x_val.shape[1], dtype=x_val.dtype)
            x_val = torch.cat([x_val, padding_points], dim=0)

            padding_y_tgt = torch.zeros(max_n_points - current_n_points, dtype=y_tgt.dtype)
            y_tgt = torch.cat([y_tgt, padding_y_tgt], dim=0)

            padding_resid = torch.zeros(max_n_points - current_n_points, dtype=resid.dtype)
            resid = torch.cat([resid, padding_resid], dim=0)

        # Padding dim维度
        if current_dim < max_dim:
            padding_dim = torch.zeros(x_val.shape[0], max_dim - current_dim, dtype=x_val.dtype)
            x_val = torch.cat([x_val, padding_dim], dim=1)

        # 创建维度mask：1表示有效维度，0表示padding
        dim_mask = torch.zeros(max_dim, dtype=torch.float32)
        dim_mask[:current_dim] = 1.0

        # 创建点mask：1表示真实点，0表示填充点
        point_mask = torch.zeros(max_n_points, dtype=torch.float32)
        point_mask[:current_n_points] = 1.0

        x_values_padded.append(x_val)          # [max_n_points, max_dim]
        y_target_padded.append(y_tgt)          # [max_n_points]  # 新增
        residuals_padded.append(resid)         # [max_n_points]
        dim_masks.append(dim_mask)             # [max_dim]
        point_masks.append(point_mask)         # [max_n_points]

    result = {
        'x_values': torch.stack(x_values_padded),       # [batch_size, max_n_points, max_dim]
        'y_target': torch.stack(y_target_padded),       # [batch_size, max_n_points]  # 新增
        'residuals': torch.stack(residuals_padded),     # [batch_size, max_n_points]  # 保留
        'dim_mask': torch.stack(dim_masks),             # [batch_size, max_dim]
        'point_mask': torch.stack(point_masks),         # [batch_size, max_n_points]
        'z0_token_ids': torch.stack([item['z0_token_ids'] for item in batch]),
        'z1_token_ids': torch.stack([item['z1_token_ids'] for item in batch]),
        'gap_token': batch[0]['gap_token'],
        'original_dims': original_dims  # 记录原始维度，供调试使用
    }

    return result


