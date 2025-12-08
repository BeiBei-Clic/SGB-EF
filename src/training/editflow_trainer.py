"""
EditFlow训练器 - 实现基于残差条件的编辑流模型训练
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer

from ..symbolic.data_generator import generate_triplet_samples
from ..modeling.condition_encoder import ConditionEncoder
from ..modeling.editflow_transformer import EditFlowTransformer, EditFlowConfig


class TripletDataset(torch.utils.data.Dataset):
    """三元组数据集 (E_curr, E_target, r, z)"""

    def __init__(self, samples: List[Dict], tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token = tokenizer.pad_token_id if tokenizer.pad_token is not None else tokenizer.eos_token_id
        self.bos_token = tokenizer.bos_token_id if tokenizer.bos_token is not None else tokenizer.cls_token_id

        # 创建特殊token映射，使用预训练模型词汇表中的token
        self.special_tokens = {
            'add': 'add', 'sub': 'sub', 'mul': 'mul', 'div': 'div', 'pow': 'pow',
            'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'exp': 'exp', 'log': 'log', 'sqrt': 'sqrt',
        }
        # 变量token
        self.max_dim = 10
        for i in range(self.max_dim):
            self.special_tokens[f'x{i}'] = f'x{i}'

    def __len__(self):
        return len(self.samples)

    def _tokenize_expression(self, tree_str: str) -> List[int]:
        """将表达式树字符串转换为token序列，使用预训练模型的tokenizer"""
        if not tree_str:
            return []

        # 构建表达式字符串
        tokens = tree_str.split(',')
        expression_tokens = []

        for token in tokens:
            if token in self.special_tokens:
                # 使用预训练模型tokenizer处理特殊token
                encoded = self.tokenizer.encode(self.special_tokens[token], add_special_tokens=False)
                expression_tokens.extend(encoded)
            elif token.replace('.', '').replace('-', '').isdigit():
                # 处理数字
                encoded = self.tokenizer.encode(token, add_special_tokens=False)
                expression_tokens.extend(encoded)

        return expression_tokens

    def __getitem__(self, idx):
        sample = self.samples[idx]

        x_values = torch.FloatTensor(sample['x_values'])
        residuals = torch.FloatTensor(sample['residuals'])
        if residuals.dim() == 1:
            residuals = residuals.unsqueeze(-1)

        curr_tokens = self._tokenize_expression(sample['tree_cur1'])
        target_tokens = self._tokenize_expression(sample['tree_gt'])

        max_len = 64
        def pad_sequence(tokens):
            tokens = [self.bos_token] + tokens[:max_len-1]
            tokens.extend([self.pad_token] * (max_len - len(tokens)))
            return torch.LongTensor(tokens)

        return {
            'x_values': x_values,
            'residuals': residuals,
            'curr_token_ids': pad_sequence(curr_tokens),
            'target_token_ids': pad_sequence(target_tokens),
            'alignment': sample['alignment_vector']['alignment']
        }


class EditFlowLoss:
    """EditFlow训练损失函数"""

    def __init__(self, scheduler_type='cubic', tokenizer=None):
        self.scheduler_type = scheduler_type
        self.tokenizer = tokenizer
        # 动态获取函数token映射，而不是硬编码
        self.func_map = self._build_function_map() if tokenizer else {}

    def _build_function_map(self):
        """使用分词器构建函数token映射"""
        function_names = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']
        func_map = {}

        for func_name in function_names:
            # 使用分词器编码函数名，获取token ID
            tokens = self.tokenizer.encode(func_name, add_special_tokens=False)
            if tokens:
                # 使用第一个token的ID作为映射
                func_map[func_name] = tokens[0]
            else:
                print(f"Warning: Function '{func_name}' not found in tokenizer vocabulary")

        return func_map

    def print_function_mapping(self):
        """打印函数映射信息，用于调试"""
        if self.func_map:
            print("函数token映射:")
            for func_name, token_id in self.func_map.items():
                token_text = self.tokenizer.decode([token_id]) if self.tokenizer else f"ID: {token_id}"
                print(f"  {func_name} -> {token_id} ('{token_text}')")
        else:
            print("警告: 没有可用的函数映射")

    def scheduler(self, t: torch.Tensor) -> torch.Tensor:
        return 3 * t**2 - 2 * t**3 if self.scheduler_type == 'cubic' else t

    def scheduler_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return 6 * t - 6 * t**2 if self.scheduler_type == 'cubic' else torch.ones_like(t)

    def alignment_to_target_mask(self, alignment: List[Tuple], seq_len: int, vocab_size: int) -> torch.Tensor:
        target_mask = torch.zeros(seq_len, 3 + 2 * vocab_size)

        for i, (op, src, tgt) in enumerate(alignment[:seq_len]):
            if op == 'keep':
                target_mask[i, 1] = 0.1
            elif op == 'insert':
                target_mask[i, 0] = 1.0
                if tgt in self.func_map and 0 <= self.func_map[tgt] < vocab_size:
                    target_mask[i, 3 + self.func_map[tgt]] = 1.0
            elif op == 'delete':
                target_mask[i, 2] = 1.0
            elif op == 'substitute':
                target_mask[i, 1] = 1.0
                if tgt in self.func_map and 0 <= self.func_map[tgt] < vocab_size:
                    target_mask[i, 3 + vocab_size + self.func_map[tgt]] = 1.0

        return target_mask

    def __call__(self, pred_rates: torch.Tensor, pred_ins_probs: torch.Tensor,
                 pred_sub_probs: torch.Tensor, alignment: List[Tuple],
                 t: torch.Tensor, vocab_size: int) -> torch.Tensor:
        batch_size, seq_len, _ = pred_rates.shape

        insert_rates = pred_rates[:, :, 0:1]
        substitute_rates = pred_rates[:, :, 1:2]
        delete_rates = pred_rates[:, :, 2:3]

        ins_probs = insert_rates * pred_ins_probs
        sub_probs = substitute_rates * pred_sub_probs
        u_cat = torch.cat([insert_rates, substitute_rates, delete_rates, ins_probs, sub_probs], dim=-1)
        u_cat = torch.clamp(u_cat, min=1e-8)

        target_masks = torch.stack([
            self.alignment_to_target_mask(align, seq_len, vocab_size)
            for align in alignment
        ]).to(pred_rates.device)

        u_total = pred_rates.sum(dim=(1, 2))
        sched_t = self.scheduler(t)
        sched_dt = self.scheduler_derivative(t)
        sched_coeff = torch.clamp((sched_dt / (1 - sched_t + 1e-8)).squeeze(-1), min=-10, max=10)

        cross_entropy = (torch.clamp(u_cat.log(), min=-20) * target_masks).sum(dim=(1, 2))
        loss = u_total - cross_entropy * sched_coeff

        return loss.mean()


def sample_conditional_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, scheduler) -> torch.Tensor:
    t = t.view(-1, 1).expand(-1, x0.size(1))
    return ((1 - scheduler(t)) * x0 + scheduler(t) * x1).long()


def custom_collate_fn(batch):
    return {
        'x_values': torch.stack([item['x_values'] for item in batch]),
        'residuals': torch.stack([item['residuals'] for item in batch]),
        'curr_token_ids': torch.stack([item['curr_token_ids'] for item in batch]),
        'target_token_ids': torch.stack([item['target_token_ids'] for item in batch]),
        'alignment': [item['alignment'] for item in batch]
    }


class EditFlowTrainer:
    """EditFlow模型训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(args.seed)

    def set_seed(self, seed: int):
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def prepare_data(self, tokenizer):
        print("生成三元组训练数据...")
        samples = generate_triplet_samples(
            num_samples=self.args.num_samples,
            max_dim=self.args.max_dim,
            n_points=self.args.n_points,
            max_depth=self.args.max_depth
        )

        dimension_groups = {}
        for sample in samples:
            dim = sample['input_dimension']
            dimension_groups.setdefault(dim, []).append(sample)

        dataloaders = {}
        datasets = {}

        for dim, dim_samples in dimension_groups.items():
            print(f"维度 {dim}: {len(dim_samples)} 个样本")
            dataset = TripletDataset(dim_samples, tokenizer)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=True,
                num_workers=0, collate_fn=custom_collate_fn
            )
            dataloaders[dim] = dataloader
            datasets[dim] = dataset

        return dataloaders, datasets, dimension_groups

    def setup_models(self):
        print("初始化tokenizer和模型...")

        # 首先初始化tokenizer，获取预训练模型的词汇表
        model_name = getattr(self.args, 'base_model_name', "openai-community/gpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 确保tokenizer有pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

        print("初始化条件编码器...")
        condition_encoder = ConditionEncoder().to(self.device)

        print("初始化EditFlow模型...")
        config = EditFlowConfig(
            max_seq_len=64,
            condition_dim=condition_encoder.output_dim,
            use_condition_injection=True,
            base_model_name=model_name
        )
        # vocab_size 将从tokenizer动态获取
        config.vocab_size = tokenizer.vocab_size
        model = EditFlowTransformer(config).to(self.device)
        print(f"EditFlow模型参数数量: {sum(p.numel() for p in model.parameters())}")

        criterion = EditFlowLoss(scheduler_type='cubic', tokenizer=tokenizer)
        criterion.print_function_mapping()  # 打印函数映射用于调试
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(condition_encoder.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        return model, condition_encoder, criterion, optimizer, tokenizer

    def train_epoch(self, model, condition_encoder, criterion, optimizer, dataloader, dataset, epoch, dimension):
        model.train()
        condition_encoder.train()

        total_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} - Dim {dimension}")

        for batch in progress_bar:
            x_values = batch['x_values'].to(self.device)
            residuals = batch['residuals'].to(self.device)
            curr_token_ids = batch['curr_token_ids'].to(self.device)
            target_token_ids = batch['target_token_ids'].to(self.device)
            alignment = batch['alignment']

            t = torch.rand(curr_token_ids.size(0), 1, device=self.device)
            condition = condition_encoder(x_values, residuals)

            import random
            x_t = curr_token_ids if random.random() < 0.5 else sample_conditional_path(
                curr_token_ids, target_token_ids, t, criterion.scheduler
            )

            attention_mask = (x_t != dataset.pad_token).float()
            pred_rates, pred_ins_probs, pred_sub_probs = model(
                input_ids=x_t, time_steps=t, condition=condition, attention_mask=attention_mask
            )

            loss = criterion(
                pred_rates=pred_rates, pred_ins_probs=pred_ins_probs, pred_sub_probs=pred_sub_probs,
                alignment=alignment, t=t, vocab_size=model.config.vocab_size
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(condition_encoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})

        return total_loss / num_batches, num_batches

    def save_checkpoint(self, model, condition_encoder, optimizer, loss, epoch, config):
        import os
        checkpoint_path = os.path.join(self.args.save_dir, f"editflow_epoch_{epoch+1}.pth")
        os.makedirs(self.args.save_dir, exist_ok=True)

        torch.save({
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
            'condition_encoder_state_dict': condition_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': loss,
            'config': config, 'args': self.args
        }, checkpoint_path)

        return checkpoint_path

    def train(self):
        print(f"使用设备: {self.device}")

        # 首先设置模型获取tokenizer
        model, condition_encoder, criterion, optimizer, tokenizer = self.setup_models()

        # 使用tokenizer准备数据
        dataloaders, datasets, dimension_groups = self.prepare_data(tokenizer)

        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"条件编码器参数数量: {sum(p.numel() for p in condition_encoder.parameters() if p.requires_grad):,}")
        print(f"开始训练 ({self.args.num_epochs} epochs)...")

        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            total_batches = 0

            for dim, dataloader in dataloaders.items():
                print(f"\n训练维度 {dim} 的数据...")
                dataset = datasets[dim]
                dim_loss, dim_batches = self.train_epoch(
                    model, condition_encoder, criterion, optimizer,
                    dataloader, dataset, epoch, dim
                )
                total_loss += dim_loss * dim_batches
                total_batches += dim_batches
                print(f"维度 {dim} 平均损失: {dim_loss:.4f}")

            avg_loss = total_loss / total_batches
            print(f"\nEpoch {epoch+1}/{self.args.num_epochs} 完成, 总体平均损失: {avg_loss:.4f}")

            if (epoch + 1) % self.args.save_every == 0:
                checkpoint_path = self.save_checkpoint(
                    model, condition_encoder, optimizer, avg_loss, epoch, model.config
                )
                print(f"检查点已保存到: {checkpoint_path}")

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