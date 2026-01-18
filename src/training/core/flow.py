"""
EditFlow连续流匹配的核心组件
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from datasets import load_dataset


def remove_gap_tokens(z_t: torch.Tensor, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """移除gap token并返回处理后的序列

    重要：此函数只移除 gap_token，保留 BOS token 和所有其他 tokens
    确保返回的序列格式为 [BOS] + [non_gap_tokens] + [PAD...]
    """
    pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
    gap_token_id = tokenizer.convert_tokens_to_ids('<gap>')
    bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
    batch_size = z_t.shape[0]
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
    batch_size = z_gap_mask.shape[0]
    x_seq_len = x_ut.shape[1]

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
        self.op_weights = None
        self._cached_weight = None
        self._cached_vocab_size = None
        self._cached_key = None

    def set_op_weights(self, op_weights):
        self.op_weights = op_weights
        self._cached_weight = None
        self._cached_vocab_size = None
        self._cached_key = None

    def make_ut_mask_from_z(self, z_t: torch.Tensor, z_1: torch.Tensor, vocab_size: int,
                           gap_token: int, tokenizer, x_t: torch.Tensor) -> torch.Tensor:
        """
        使用明确的位置映射表生成正确的编辑操作掩码

        核心改进：构建Z空间到X空间的明确映射表，避免双索引遍历导致的位置错位问题

        Args:
            z_t: 当前状态（Z空间，含gap）[batch, z_seq_len]
            z_1: 目标状态（Z空间，含gap）[batch, z_seq_len]
            vocab_size: 词汇表大小
            gap_token: gap token的ID
            tokenizer: 分词器
            x_t: 当前状态（X空间，无gap）[batch, x_seq_len]

        Returns:
            u_mask: 编辑操作掩码 [batch, x_seq_len, 2*vocab_size+2]
                    维度布局：[INS(vocab_size) | DEL(1) | SUB(vocab_size) | KEEP(1)]
        """
        batch_size, z_seq_len = z_t.shape
        x_seq_len = x_t.shape[1]
        n_ops = 2 * vocab_size + 2  # 插入(vocab_size) + 删除(1) + 替换(vocab_size) + KEEP(1)

        pad_token = tokenizer.convert_tokens_to_ids('<pad>')
        bos_token = tokenizer.convert_tokens_to_ids('<s>')

        # 初始化输出掩码（在X空间）
        u_mask = torch.zeros((batch_size, x_seq_len, n_ops), dtype=torch.int, device=z_t.device)

        # 对每个样本进行处理
        for b in range(batch_size):
            # === 步骤1：构建Z空间到X空间的位置映射表 ===
            z_to_x_map = {}  # {z_pos: x_pos or None}
            insert_positions = []  # 记录所有gap位置

            x_index = 0
            for z_pos in range(z_seq_len):
                token_t = z_t[b, z_pos].item()

                if token_t == pad_token:
                    z_to_x_map[z_pos] = None
                    continue

                # BOS映射到X空间位置0，确保gap可以找到BOS作为插入点
                if token_t == bos_token:
                    z_to_x_map[z_pos] = x_index
                    x_index += 1
                    continue

                if token_t != gap_token:
                    z_to_x_map[z_pos] = x_index
                    x_index += 1
                else:
                    # gap位置：记录为插入点
                    z_to_x_map[z_pos] = None
                    insert_positions.append(z_pos)

            # === 步骤2：处理非gap位置的操作（SUBSTITUTE/DELETE/KEEP） ===
            for z_pos in range(z_seq_len):
                token_t = z_t[b, z_pos].item()
                token_1 = z_1[b, z_pos].item()

                if token_t == pad_token or token_1 == pad_token:
                    continue

                if token_t == gap_token:
                    continue  # gap位置的INSERT操作在步骤3处理

                x_pos = z_to_x_map[z_pos]
                if x_pos is None or x_pos >= x_seq_len:
                    continue

                # 判断操作类型
                if token_1 == gap_token:
                    u_mask[b, x_pos, vocab_size] = 1  # DELETE
                elif token_t != token_1:
                    u_mask[b, x_pos, token_1 + vocab_size + 1] = 1  # SUBSTITUTE
                else:
                    u_mask[b, x_pos, -1] = 1  # KEEP

            # === 步骤3：处理gap位置的INSERT操作 ===
            for gap_z_pos in insert_positions:
                token_t = z_t[b, gap_z_pos].item()
                token_1 = z_1[b, gap_z_pos].item()

                if token_t == pad_token or token_1 == pad_token:
                    continue

                if token_t == gap_token and token_1 != gap_token:
                    # INSERT操作：在gap之前的第一个非gap位置之后插入token_1
                    insert_x_pos = None
                    for prev_z_pos in range(gap_z_pos - 1, -1, -1):
                        if z_to_x_map[prev_z_pos] is not None:
                            insert_x_pos = z_to_x_map[prev_z_pos]
                            break

                    # 如果gap之前没有非gap位置，插入到开头
                    if insert_x_pos is None:
                        insert_x_pos = 0

                    # 标记INSERT操作（INSERT优先级高于KEEP）
                    if 0 <= insert_x_pos < x_seq_len:
                        if u_mask[b, insert_x_pos, -1].item() == 1:
                            u_mask[b, insert_x_pos, -1] = 0  # 移除KEEP
                        u_mask[b, insert_x_pos, token_1] = 1  # INSERT

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

        # 获取操作空间维度
        _, _, n_ops = u_z.shape

        # 使用标准cross_entropy计算loss
        # 有了KEEP操作后，每个token位置都有一个明确的操作标签：
        # - 需要编辑的位置：标记 ins/del/sub
        # - 不需要编辑的位置：标记 KEEP
        # - 只有padding位置无标签
        #
        # 因此可以安全地使用标准cross_entropy，无需复杂的掩码逻辑

        # 步骤1: 将one-hot编码的u_mask转换为标签索引
        # u_mask: [batch, z_seq_len, n_ops] -> target_ids: [batch, z_seq_len]
        target_ids = u_mask.argmax(dim=-1)

        # 步骤2: 计算标准cross_entropy（对每个token位置）
        # u_z: [batch, z_seq_len, n_ops] -> [batch*z_seq_len, n_ops]
        # target_ids: [batch, z_seq_len] -> [batch*z_seq_len]
        weight = None
        if self.op_weights is not None:
            key = (vocab_size, self.op_weights["ins"], self.op_weights["del"],
                   self.op_weights["sub"], self.op_weights["keep"])
            if self._cached_weight is None or self._cached_key != key:
                n_ops = 2 * vocab_size + 2
                weight = torch.ones(n_ops, dtype=torch.float32, device=u_z.device)
                weight[:vocab_size] *= self.op_weights["ins"]
                weight[vocab_size] *= self.op_weights["del"]
                weight[vocab_size + 1:2 * vocab_size + 1] *= self.op_weights["sub"]
                weight[2 * vocab_size + 1] *= self.op_weights["keep"]
                self._cached_weight = weight
                self._cached_vocab_size = vocab_size
                self._cached_key = key
            else:
                weight = self._cached_weight

        loss_per_token = F.cross_entropy(
            u_z.reshape(-1, n_ops),
            target_ids.reshape(-1),
            weight=weight,
            reduction='none'  # 先不归一化，后续手动处理
        )  # [batch*z_seq_len]

        # 步骤3: 过滤padding位置（使用u_mask判断）
        # 如果某个位置所有操作都是0，说明是padding
        valid_positions_mask = (u_mask.sum(dim=-1) > 0)  # [batch, z_seq_len]
        valid_positions_mask_flat = valid_positions_mask.reshape(-1)  # [batch*z_seq_len]

        # 只对有效位置计算loss
        loss_per_token = loss_per_token[valid_positions_mask_flat]

        # 步骤4: 按样本归一化（避免长序列主导loss）
        # 记录每个样本的有效token数
        valid_tokens_per_sample = valid_positions_mask.sum(dim=1)  # [batch]
        valid_tokens_per_sample = valid_tokens_per_sample.clamp(min=1)  # 避免除0

        # 计算每个样本的平均loss
        sample_losses = []
        start_idx = 0
        for num_tokens in valid_tokens_per_sample:
            end_idx = start_idx + num_tokens.item()
            sample_losses.append(loss_per_token[start_idx:end_idx].mean())
            start_idx = end_idx

        cross_entropy = torch.stack(sample_losses)  # [batch]

        # 最终损失：交叉熵损失
        return cross_entropy.mean()


def prepare_dataset_hf(data_file: str, tokenizer, max_expr_length: int = 128,
                       stream: bool = True, num_proc: Optional[int] = None,
                       skip: Optional[int] = None, take: Optional[int] = None,
                       logger=None):
    """
    使用 Hugging Face datasets 加载预处理好的数据

    注意：数据生成时已完成 BOS 添加和 Padding，直接加载使用

    Args:
        data_file: 数据文件路径 (.parquet格式)
        tokenizer: 分词器
        max_expr_length: 表达式最大长度（仅用于向后兼容，数据已预先处理）
        stream: 是否使用流式加载（默认True，适合大文件）
        num_proc: 预处理时的进程数（已废弃，保留仅为兼容）
        skip: 跳过前N个样本
        take: 读取N个样本后停止
        logger: 日志记录器（可选，如果提供则使用logger而非print）

    Returns:
        dataset: Hugging Face Dataset 对象（直接返回，不使用FlowDataset封装）
    """
    import time
    data_files = {"train": data_file}

    # 加载Parquet格式数据
    load_start = time.time()
    if stream:
        # 流式加载：适合大文件，不一次性加载到内存
        raw_dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    else:
        # 一次性加载：适合小文件，后续处理更快
        raw_dataset = load_dataset("parquet", data_files=data_files, split="train")
    load_time = time.time() - load_start

    if logger:
        logger.log("DATA_LOAD", f"load_dataset() 耗时: {load_time:.2f}秒", "data_loading", level=1)
    else:
        print(f"  [性能] load_dataset() 耗时: {load_time:.2f}秒")

    # 应用skip和take
    if skip is not None:
        raw_dataset = raw_dataset.skip(skip)
    if take is not None:
        raw_dataset = raw_dataset.take(take)

    # 添加 gap_token 列（数据生成时没有这个列）
    gap_token_id = tokenizer.convert_tokens_to_ids('<gap>')

    def add_gap_token(examples):
        """为每个样本添加 gap_token 常量"""
        batch_size = len(examples['x_values'])
        return {'gap_token': [gap_token_id] * batch_size}

    add_gap_start = time.time()
    map_kwargs = {"batched": True}
    if not stream:
        map_kwargs["num_proc"] = 1  # 单进程即可，只是添加常量
    tokenized_dataset = raw_dataset.map(add_gap_token, **map_kwargs)
    add_gap_time = time.time() - add_gap_start

    if logger:
        logger.log("DATA_GAP_TOKEN", f"添加 gap_token 列耗时: {add_gap_time:.2f}秒", "data_loading", level=1)
    else:
        print(f"  [性能] 数据已预先 padded，跳过 map() 操作，仅添加 gap_token 列...")
        print(f"  [性能] 添加 gap_token 列耗时: {add_gap_time:.2f}秒")

    # 添加 shuffle
    buffer_size_info = f"buffer_size={10000 if stream else '全量'}"
    shuffle_start = time.time()
    if stream:
        tokenized_dataset = tokenized_dataset.shuffle(seed=42, buffer_size=10000)
    else:
        tokenized_dataset = tokenized_dataset.shuffle(seed=42)
    shuffle_time = time.time() - shuffle_start

    if logger:
        logger.log("DATA_SHUFFLE", f"shuffle操作完成 ({buffer_size_info}) | 耗时: {shuffle_time:.2f}秒", "data_loading", level=1)
    else:
        print(f"  [性能] 开始shuffle操作 ({buffer_size_info})...")
        print(f"  [性能] shuffle操作完成，耗时: {shuffle_time:.2f}秒")

    # 设置格式为torch
    format_start = time.time()
    if stream:
        tokenized_dataset = tokenized_dataset.with_format(type='torch')
    else:
        tokenized_dataset.set_format(type='torch', columns=[
            'x_values', 'y_target', 'residuals',
            'z0_token_ids', 'z1_token_ids', 'gap_token'
        ])
    format_time = time.time() - format_start

    if logger:
        logger.log("DATA_FORMAT", f"格式设置完成 | 耗时: {format_time:.2f}秒", "data_loading", level=1)
    else:
        print(f"  [性能] 开始设置torch格式...")
        print(f"  [性能] 格式设置完成，耗时: {format_time:.2f}秒")

    # 保存tokenizer引用供后续使用
    tokenized_dataset.tokenizer = tokenizer

    # 添加必要的属性（替代FlowDataset的属性）
    tokenized_dataset.stream = stream

    # 对于流式数据集，添加set_epoch方法
    if stream:
        def set_epoch_method(epoch):
            """设置epoch号，用于重置迭代器并重新洗牌"""
            pass  # Hugging Face流式数据集会自动处理
        tokenized_dataset.set_epoch = set_epoch_method

    return tokenized_dataset


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

    for item in batch:
        x_val = item['x_values']  # [n_points, current_dim]
        y_tgt = item['y_target']  # [n_points]

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
    y_target_padded = []
    residuals_padded = []
    dim_masks = []
    point_masks = []

    for item in batch:
        x_val = item['x_values'].clone()  # [n_points, current_dim]
        y_tgt = item['y_target'].clone()  # [n_points]
        resid = item['residuals'].clone()  # [n_points]

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
        y_target_padded.append(y_tgt)          # [max_n_points]
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
        'original_dims': original_dims,  # 记录原始维度，供调试使用
    }

    return result
