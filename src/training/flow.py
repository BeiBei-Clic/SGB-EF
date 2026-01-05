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
        loss_per_token = F.cross_entropy(
            u_z.reshape(-1, n_ops),
            target_ids.reshape(-1),
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
                       skip: Optional[int] = None, take: Optional[int] = None):
    """
    使用 Hugging Face datasets 加载并预处理数据

    Args:
        data_file: 数据文件路径 (.parquet格式)
        tokenizer: 分词器
        max_expr_length: 表达式最大长度
        stream: 是否使用流式加载（默认True，适合大文件）
        num_proc: 预处理时的进程数，None表示自动选择
        skip: 跳过前N个样本（在map之前应用）
        take: 读取N个样本后停止（在map之前应用）

    Returns:
        dataset: Hugging Face Dataset 对象
    """
    data_files = {"train": data_file}

    # 加载Parquet格式数据
    if stream:
        # 流式加载：适合大文件，不一次性加载到内存
        raw_dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    else:
        # 一次性加载：适合小文件，后续处理更快
        raw_dataset = load_dataset("parquet", data_files=data_files, split="train")

    # 在map之前应用skip和take（更可靠，避免在已处理的数据集上操作）
    if skip is not None:
        raw_dataset = raw_dataset.skip(skip)
    if take is not None:
        raw_dataset = raw_dataset.take(take)

    # 获取token相关信息
    pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
    bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
    gap_token_id = tokenizer.convert_tokens_to_ids('<gap>')

    def process_function(examples):
        """
        预处理函数：将Parquet数据转换为模型所需的格式
        Parquet直接返回字典，不需要json.loads
        """
        # Parquet加载后直接是字典列表
        # 获取batch size
        if isinstance(examples['x_values'], list):
            batch_size = len(examples['x_values'])
        else:
            # 单个样本的情况
            batch_size = 1
            examples = {k: [v] for k, v in examples.items()}

        # 预分配列表
        outputs = {
            'x_values': examples['x_values'],
            'y_target': examples['y_target'],
            'residuals': examples['residuals'],
            'z0_token_ids': [],
            'z1_token_ids': [],
            'gap_token': []
        }

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

        # 直接使用预计算的token IDs（需要逐个处理因为需要padding）
        for i in range(batch_size):
            z0_token_ids = examples['z0_token_ids'][i]
            z1_token_ids = examples['z1_token_ids'][i]

            outputs['z0_token_ids'].append(pad_z_sequence(z0_token_ids))
            outputs['z1_token_ids'].append(pad_z_sequence(z1_token_ids))
            outputs['gap_token'].append(gap_token_id)

        return outputs

    # 应用预处理
    if stream:
        # 流式模式：使用map (IterableDataset不支持desc参数)
        tokenized_dataset = raw_dataset.map(
            process_function,
            batched=True,
            remove_columns=['x_values', 'y_target', 'residuals', 'z0_tokens', 'z1_tokens']
        )
        # 添加 shuffle 以提高数据多样性和支持 set_epoch
        tokenized_dataset = tokenized_dataset.shuffle(seed=42, buffer_size=10000)
    else:
        # 非流式模式：使用多进程加速
        tokenized_dataset = raw_dataset.map(
            process_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=['x_values', 'y_target', 'residuals', 'z0_tokens', 'z1_tokens'],
            desc="Preprocessing dataset"
        )
        # 非流式模式也添加 shuffle
        tokenized_dataset = tokenized_dataset.shuffle(seed=42)

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
                 num_proc: Optional[int] = None, logger=None,
                 skip: Optional[int] = None, take: Optional[int] = None):
        """
        使用 Hugging Face datasets 的数据集包装器

        Args:
            data_file: 数据文件路径
            tokenizer: 分词器
            max_dim: 最大维度（保留用于向后兼容）
            max_expr_length: 表达式最大长度
            stream: 是否使用流式加载（默认True）
            num_proc: 预处理时的进程数
            logger: 日志记录器
            skip: 跳过前N个样本（用于test数据集）
            take: 读取N个样本后停止（用于train/test分割）
        """
        self.tokenizer = tokenizer
        self.max_dim = max_dim
        self.max_expr_length = max_expr_length
        self.vocab_size = len(tokenizer.get_vocab())
        self.pad_token = tokenizer.convert_tokens_to_ids('<pad>')
        self.bos_token = tokenizer.convert_tokens_to_ids('<s>')
        self.gap_token = tokenizer.convert_tokens_to_ids('<gap>')
        self.stream = stream  # 保存stream模式标志
        self.logger = logger  # 保存logger引用
        self._skip = skip  # 保存skip参数
        self._take = take  # 保存take参数

        # 保存数据集参数，用于后续重新加载
        self._data_file = data_file
        self._num_proc = num_proc

        # 使用 Hugging Face datasets 加载数据
        self._hf_dataset = prepare_dataset_hf(
            data_file=data_file,
            tokenizer=tokenizer,
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
            # 流式模式：从parquet元数据获取行数（快速准确）
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(data_file)
                self._dataset_length = pf.metadata.num_rows
            except Exception as e:
                # 如果parquet读取失败，尝试遍历dataset（较慢）
                print(f"警告: 无法从Parquet元数据获取行数，尝试遍历dataset。错误: {e}")
                try:
                    self._dataset_length = sum(1 for _ in self._hf_dataset)
                except:
                    # 如果遍历也失败，使用默认值
                    print(f"警告: 无法统计数据集行数，使用默认值。")
                    self._dataset_length = 1000000  # 默认值

    def __len__(self):
        """返回数据集大小（考虑skip和take限制）"""
        if self._take is not None:
            # 如果设置了take，返回take值（最多不超过数据集总大小）
            return min(self._take, self._dataset_length)
        return self._dataset_length

    def __iter__(self):
        """流式模式下的迭代器"""
        mode = "非流式" if self._data_list is not None else "流式"
        if self.logger:
            self.logger.log(
                "DATA_ITER_START",
                f"创建数据迭代器 | 模式={mode} | 数据集大小={self._dataset_length} | "
                f"skip={self._skip} | take={self._take}",
                "data_loading",
                level=1
            )

        if self._data_list is not None:
            # 非流式模式：迭代缓存列表（应用skip和take）
            start_idx = self._skip if self._skip is not None else 0
            end_idx = (start_idx + self._take) if self._take is not None else None
            return iter(self._data_list[start_idx:end_idx])
        else:
            # 流式模式：每次重新创建Hugging Face dataset以确保迭代器可重用
            # skip和take在prepare_dataset_hf内部的map之前应用，更可靠
            hf_dataset = prepare_dataset_hf(
                data_file=self._data_file,
                tokenizer=self.tokenizer,
                max_expr_length=self.max_expr_length,
                stream=True,
                num_proc=self._num_proc,
                skip=self._skip,
                take=self._take
            )
            return hf_dataset.__iter__()

    def set_epoch(self, epoch):
        """设置 epoch 号，用于重置迭代器并重新洗牌（Hugging Face datasets 标准方法）

        Args:
            epoch: 当前 epoch 号（从0开始）
        """
        # 直接调用底层 Hugging Face dataset 的 set_epoch 方法
        if hasattr(self._hf_dataset, 'set_epoch'):
            self._hf_dataset.set_epoch(epoch)
        else:
            # 如果 _hf_dataset 不支持 set_epoch（已经是列表形式），则不需要处理
            pass

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
        'original_dims': original_dims  # 记录原始维度，供调试使用
    }

    return result


