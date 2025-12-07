"""
EditFlow训练器 - 实现基于残差条件的编辑流模型训练
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

from ..symbolic.data_generator import generate_triplet_samples
from ..modeling.condition_encoder import ConditionEncoder
from ..modeling.editflow_transformer import EditFlowTransformer, EditFlowConfig


class TripletDataset(torch.utils.data.Dataset):
    """三元组数据集 (E_curr, E_target, r, z)"""

    def __init__(self, samples: List[Dict], vocab_size: int = 1000):
        self.samples = samples
        self.vocab_size = vocab_size
        self.pad_token = vocab_size
        self.bos_token = vocab_size + 1
        self.max_dim = 10  # 支持最高十维变量

    def __len__(self):
        return len(self.samples)

    def _build_token_mapping(self) -> Dict[str, int]:
        """动态构建token映射，支持最高十维变量"""
        token_to_id = {
            'add': 1, 'sub': 2, 'mul': 3, 'div': 4, 'pow': 5,
            'sin': 6, 'cos': 7, 'tan': 8, 'exp': 9, 'log': 10, 'sqrt': 11,
        }

        # 动态添加变量token x0-x9
        for i in range(self.max_dim):
            token_to_id[f'x{i}'] = 12 + i

        return token_to_id

    def _tokenize_expression(self, tree_str: str) -> List[int]:
        """将表达式树字符串转换为token序列"""
        if not tree_str:
            return []

        tokens = tree_str.split(',')
        token_to_id = self._build_token_mapping()

        token_ids = []
        for token in tokens:
            if token in token_to_id:
                token_ids.append(token_to_id[token])
            elif token.replace('.', '').replace('-', '').isdigit():
                # 常数映射
                const_val = float(token)
                const_id = 50 + min(int(abs(const_val)) % 50, 49)
                token_ids.append(const_id)
            else:
                token_ids.append(200)  # 未知token

        return token_ids

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 提取数据
        x_values = torch.FloatTensor(sample['x_values'])
        residuals = torch.FloatTensor(sample['residuals'])

        # Token化表达式
        curr_tokens = self._tokenize_expression(sample['tree_cur1'])
        target_tokens = self._tokenize_expression(sample['tree_gt'])

        # 添加BOS token并填充
        max_len = 64  # 固定最大长度

        def pad_sequence(tokens, max_len):
            tokens = [self.bos_token] + tokens[:max_len-1]
            if len(tokens) < max_len:
                tokens.extend([self.pad_token] * (max_len - len(tokens)))
            return torch.LongTensor(tokens)

        curr_token_ids = pad_sequence(curr_tokens, max_len)
        target_token_ids = pad_sequence(target_tokens, max_len)

        # 对齐信息
        alignment = sample['alignment_vector']['alignment']

        return {
            'x_values': x_values,
            'residuals': residuals,
            'curr_token_ids': curr_token_ids,
            'target_token_ids': target_token_ids,
            'alignment': alignment
        }


class EditFlowLoss:
    """EditFlow训练损失函数"""

    def __init__(self, scheduler_type='cubic'):
        self.scheduler_type = scheduler_type

    def scheduler(self, t: torch.Tensor) -> torch.Tensor:
        """时间调度器 κ(t)"""
        if self.scheduler_type == 'cubic':
            return 3 * t**2 - 2 * t**3
        elif self.scheduler_type == 'linear':
            return t
        else:
            return t

    def scheduler_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """调度器导数 κ'(t)"""
        if self.scheduler_type == 'cubic':
            return 6 * t - 6 * t**2
        elif self.scheduler_type == 'linear':
            return torch.ones_like(t)
        else:
            return torch.ones_like(t)

    def _get_function_token_id(self, func_name: str) -> int:
        """获取函数token ID"""
        func_map = {'sin': 6, 'cos': 7, 'tan': 8, 'exp': 9, 'log': 10, 'sqrt': 11}
        return func_map.get(func_name, 6)  # 默认返回sin

    def alignment_to_target_mask(self, alignment: List[Tuple], seq_len: int, vocab_size: int) -> torch.Tensor:
        """将对齐序列转换为目标掩码"""
        # 创建目标掩码: (seq_len, 3 + 2*vocab_size)
        # [insert_rate, substitute_rate, delete_rate, insert_probs..., substitute_probs...]
        target_mask = torch.zeros(seq_len, 3 + 2 * vocab_size)

        for i, (op, src, tgt) in enumerate(alignment[:seq_len]):
            if op == 'keep':
                target_mask[i, 1] = 0.1  # 低替换率
            elif op == 'insert':
                target_mask[i, 0] = 1.0  # 高插入率
                # 设置插入的目标token
                if tgt in ['sin', 'cos', 'exp', 'log', 'sqrt']:
                    token_id = self._get_function_token_id(tgt)
                    target_mask[i, 3 + token_id] = 1.0
            elif op == 'delete':
                target_mask[i, 2] = 1.0  # 高删除率
            elif op == 'substitute':
                target_mask[i, 1] = 1.0  # 高替换率
                # 设置替换的目标token
                if tgt in ['sin', 'cos', 'exp', 'log', 'sqrt']:
                    token_id = self._get_function_token_id(tgt)
                    target_mask[i, 3 + vocab_size + token_id] = 1.0

        return target_mask

    def __call__(self, pred_rates: torch.Tensor, pred_ins_probs: torch.Tensor,
                 pred_sub_probs: torch.Tensor, alignment: List[Tuple],
                 t: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """
        计算EditFlow损失
        Args:
            pred_rates: (batch_size, seq_len, 3) 预测的编辑速率
            pred_ins_probs: (batch_size, seq_len, vocab_size) 预测的插入概率
            pred_sub_probs: (batch_size, seq_len, vocab_size) 预测的替换概率
            alignment: 对齐序列列表
            t: (batch_size, 1) 时间步
            vocab_size: 词汇表大小
        """
        batch_size, seq_len, _ = pred_rates.shape

        # 构建组合预测 u_cat
        ins_rates = pred_rates[:, :, 0:1] * pred_ins_probs  # (batch_size, seq_len, vocab_size)
        sub_rates = pred_rates[:, :, 1:2] * pred_sub_probs  # (batch_size, seq_len, vocab_size)
        del_rates = pred_rates[:, :, 2:3]  # (batch_size, seq_len, 1)

        u_cat = torch.cat([ins_rates, sub_rates, del_rates], dim=-1)  # (batch_size, seq_len, 2*vocab_size + 1)

        # 构建目标掩码
        target_masks = []
        for i, align in enumerate(alignment):
            target_mask = self.alignment_to_target_mask(align, seq_len, vocab_size)
            target_masks.append(target_mask)
        target_masks = torch.stack(target_masks).to(pred_rates.device)

        # 计算总速率
        u_total = pred_rates.sum(dim=(1, 2))  # (batch_size,)

        # 计算调度器系数
        sched_coeff = (self.scheduler_derivative(t) / (1 - self.scheduler(t) + 1e-8)).squeeze(-1)

        # 计算交叉熵项
        log_u_cat = torch.clamp(u_cat.log(), min=-20)
        cross_entropy = (log_u_cat * target_masks).sum(dim=(1, 2))

        # Bregman散度损失
        loss = u_total - cross_entropy * sched_coeff

        return loss.mean()


def sample_conditional_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, scheduler) -> torch.Tensor:
    """从x0到x1的条件路径采样"""
    # 简化的线性插值
    t = t.view(-1, 1, 1)
    return ((1 - scheduler(t)) * x0 + scheduler(t) * x1).long()


class EditFlowTrainer:
    """EditFlow模型训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(args.seed)

    def set_seed(self, seed: int):
        """设置随机种子"""
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def prepare_data(self):
        """准备训练数据"""
        print("生成三元组训练数据...")
        samples = generate_triplet_samples(
            num_samples=self.args.num_samples,
            max_dim=self.args.max_dim,
            n_points=self.args.n_points,
            max_depth=self.args.max_depth
        )

        dataset = TripletDataset(samples, vocab_size=self.args.vocab_size)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2
        )

        return dataloader, dataset

    def setup_models(self):
        """设置模型和损失函数"""
        print("初始化条件编码器...")
        # 初始化条件编码器
        condition_encoder = ConditionEncoder().to(self.device)
        print(f"初始化EditFlow模型...")
        # 初始化EditFlow模型
        config = EditFlowConfig(
            vocab_size=self.args.vocab_size + 2,  # +2 for PAD and BOS
            hidden_dim=self.args.hidden_dim,
            num_layers=self.args.num_layers,
            num_heads=self.args.num_heads,
            max_seq_len=64,
            condition_dim=condition_encoder.output_dim,
            use_condition_injection=True
        )
        model = EditFlowTransformer(config).to(self.device)
        print(f"EditFlow模型参数数量: {sum(p.numel() for p in model.parameters())}")

        # 初始化损失函数
        criterion = EditFlowLoss(scheduler_type='cubic')

        # 初始化优化器
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(condition_encoder.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        return model, condition_encoder, criterion, optimizer

    def train_epoch(self, model, condition_encoder, criterion, optimizer, dataloader, dataset, epoch):
        """训练一个epoch"""
        model.train()
        condition_encoder.train()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

        for batch in progress_bar:
            # 提取数据
            x_values = batch['x_values'].to(self.device)  # (batch_size, n_points)
            residuals = batch['residuals'].to(self.device)  # (batch_size, n_points)
            curr_token_ids = batch['curr_token_ids'].to(self.device)  # (batch_size, seq_len)
            target_token_ids = batch['target_token_ids'].to(self.device)  # (batch_size, seq_len)
            alignment = batch['alignment']  # List of alignment sequences

            # 采样时间步
            t = torch.rand(curr_token_ids.size(0), 1, device=self.device)

            # 条件编码：编码残差
            condition = condition_encoder(x_values, residuals)  # (batch_size, condition_dim)

            # 生成插值序列 (简化的条件路径)
            import random
            if random.random() < 0.5:  # 50%概率使用curr，50%使用插值
                x_t = curr_token_ids
            else:
                x_t = sample_conditional_path(curr_token_ids, target_token_ids, t, criterion.scheduler)

            # 创建attention mask
            attention_mask = (x_t != dataset.pad_token).float()

            # 前向传播
            pred_rates, pred_ins_probs, pred_sub_probs = model(
                input_ids=x_t,
                time_steps=t,
                condition=condition,
                attention_mask=attention_mask
            )

            # 计算损失
            loss = criterion(
                pred_rates=pred_rates,
                pred_ins_probs=pred_ins_probs,
                pred_sub_probs=pred_sub_probs,
                alignment=alignment,
                t=t,
                vocab_size=self.args.vocab_size
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(condition_encoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })

        return total_loss / num_batches

    def save_checkpoint(self, model, condition_encoder, optimizer, loss, epoch, config):
        """保存检查点"""
        import os
        checkpoint_path = os.path.join(self.args.save_dir, f"editflow_epoch_{epoch+1}.pth")
        os.makedirs(self.args.save_dir, exist_ok=True)

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'condition_encoder_state_dict': condition_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config,
            'args': self.args
        }, checkpoint_path)

        return checkpoint_path

    def train(self):
        """完整的训练流程"""
        print(f"使用设备: {self.device}")

        # 准备数据
        dataloader, dataset = self.prepare_data()

        # 设置模型
        model, condition_encoder, criterion, optimizer = self.setup_models()

        print(f"模型参数数量: {model.num_parameters():,}")
        print(f"条件编码器参数数量: {sum(p.numel() for p in condition_encoder.parameters() if p.requires_grad):,}")

        # 训练循环
        print(f"开始训练 ({self.args.num_epochs} epochs)...")

        for epoch in range(self.args.num_epochs):
            avg_loss = self.train_epoch(
                model, condition_encoder, criterion, optimizer,
                dataloader, dataset, epoch
            )

            print(f"Epoch {epoch+1}/{self.args.num_epochs} 完成, 平均损失: {avg_loss:.4f}")

            # 保存检查点
            if (epoch + 1) % self.args.save_every == 0:
                checkpoint_path = self.save_checkpoint(
                    model, condition_encoder, optimizer, avg_loss,
                    epoch, model.config
                )
                print(f"检查点已保存到: {checkpoint_path}")

        # 保存最终模型
        import os
        final_model_path = os.path.join(self.args.save_dir, "editflow_final.pth")
        os.makedirs(self.args.save_dir, exist_ok=True)

        torch.save({
            'epoch': self.args.num_epochs,
            'model_state_dict': model.state_dict(),
            'condition_encoder_state_dict': condition_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model.config,
            'args': self.args
        }, final_model_path)

        print(f"最终模型已保存到: {final_model_path}")

        return model, condition_encoder