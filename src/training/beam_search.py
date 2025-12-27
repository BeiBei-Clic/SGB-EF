"""
束搜索模块 - 用于符号回归推理的束搜索算法
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import heapq


@dataclass(order=True)
class BeamCandidate:
    """束搜索候选

    Attributes:
        score: 累积分数（越高越好）
        tokens: 表达式token列表
        condition: 对应的条件嵌入
        residuals: 对应的残差
        history: 操作历史（可选，用于调试）
    """
    score: float
    tokens: List[str] = field(compare=False)
    condition: Optional[torch.Tensor] = field(default=None, compare=False)
    residuals: Optional[np.ndarray] = field(default=None, compare=False)
    history: List[str] = field(default_factory=list, compare=False)

    def __repr__(self):
        return f"BeamCandidate(score={self.score:.4f}, tokens={','.join(self.tokens) if self.tokens else '<blank>'})"


@dataclass
class ActionProposal:
    """操作提案

    Attributes:
        action_type: 操作类型 ('insert', 'substitute', 'delete')
        position: 位置索引
        token: 要插入或替换的token（仅对insert/substitute有效）
        score: 操作分数
        new_tokens: 执行操作后的新token列表
    """
    action_type: str
    position: int
    token: Optional[str] = None
    score: float = 0.0
    new_tokens: List[str] = field(default_factory=list)

    def __repr__(self):
        if self.action_type == 'insert':
            return f"Insert(pos={self.position}, token={self.token}, score={self.score:.4f})"
        elif self.action_type == 'substitute':
            return f"Substitute(pos={self.position}, {self.token}, score={self.score:.4f})"
        elif self.action_type == 'delete':
            return f"Delete(pos={self.position}, score={self.score:.4f})"
        return f"Action({self.action_type}, score={self.score:.4f})"


class BeamSearchSymbolicRegression:
    """基于束搜索的符号回归推理器"""

    def __init__(self,
                 model,
                 condition_encoder,
                 tokenizer,
                 special_tokens_manager,
                 device,
                 args,
                 logger,
                 min_action_score=0.01,
                 max_expression_length=50,
                 numerical_clip_threshold=1e6):
        """初始化束搜索推理器

        Args:
            model: EditFlow模型
            condition_encoder: 条件编码器
            tokenizer: 分词器
            special_tokens_manager: 特殊token管理器
            device: 计算设备
            args: 参数配置
            logger: 日志记录器
            min_action_score: 最小操作分数阈值
            max_expression_length: 表达式最大长度
            numerical_clip_threshold: 数值裁剪阈值
        """
        self.model = model
        self.condition_encoder = condition_encoder
        self.tokenizer = tokenizer
        self.special_tokens_manager = special_tokens_manager
        self.device = device
        self.args = args
        self.logger = logger
        self.min_action_score = min_action_score
        self.max_expression_length = max_expression_length
        self.numerical_clip_threshold = numerical_clip_threshold

    def generate_action_proposals(self,
                                  current_tokens: List[str],
                                  condition: torch.Tensor,
                                  x_values: torch.Tensor,
                                  t: float,
                                  top_k: Optional[int] = None,
                                  valid_variables: Optional[List[str]] = None) -> List[ActionProposal]:
        """为当前表达式生成操作提案

        Args:
            current_tokens: 当前token列表
            condition: 条件嵌入
            x_values: 输入x值
            t: 时间步
            top_k: 每种操作类型保留的top-k数量，None表示全部
            valid_variables: 有效的变量token列表（如 ['x0', 'x1']），用于过滤无效变量

        Returns:
            操作提案列表
        """
        proposals = []

        # 构建模型输入 - 确保与训练时的序列格式完全一致
        tokenized_expr = self.special_tokens_manager.tokenize_expression(','.join(current_tokens))
        max_len = getattr(self.args, 'max_expr_length', 128)

        if len(tokenized_expr) > max_len - 1:
            tokenized_expr = tokenized_expr[:max_len-1]

        bos_token = self.special_tokens_manager.tokenizer.convert_tokens_to_ids('<s>')
        pad_token = self.special_tokens_manager.tokenizer.convert_tokens_to_ids('<pad>')

        # 关键：必须添加 BOS token，与训练时的格式一致 [BOS] + tokens + [PAD]
        tokenized_expr = [bos_token] + tokenized_expr
        tokenized_expr = tokenized_expr + [pad_token] * (max_len - len(tokenized_expr))

        # 验证 BOS token 在正确位置
        assert tokenized_expr[0] == bos_token, f"BOS token 必须在位置0，但得到 {tokenized_expr[0]}"

        input_ids = torch.LongTensor([tokenized_expr]).to(self.device)
        attention_mask = (input_ids != pad_token).float().to(self.device)
        t_tensor = torch.tensor([[t]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            rates, insert_probs, substitute_probs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                time_steps=t_tensor,
                condition=condition
            )

            lambda_ins = rates[0, :, 0].cpu().numpy()
            lambda_sub = rates[0, :, 1].cpu().numpy()
            lambda_del = rates[0, :, 2].cpu().numpy()

            # 调试：验证序列格式
            base_length = int(attention_mask[0].sum().item())
            effective_length = max(base_length, min(10, input_ids.size(1)))

            # 调试：打印序列格式信息（仅在第一次调用时）
            if not hasattr(self, '_sequence_format_logged'):
                if self.logger and self._is_main_process():
                    self.logger.log("SEQUENCE_FORMAT",
                                   f"推理序列格式验证 | input_ids[0:5]={input_ids[0, :5].tolist()} | "
                                   f"base_length={base_length} | effective_length={effective_length} | "
                                   f"current_tokens={current_tokens[:3]}",
                                   "beam_search", level=2)
                self._sequence_format_logged = True

            # 生成insert操作提案
            seq_len = min(effective_length, lambda_ins.shape[0])
            for pos in range(1, seq_len):
                if pos < lambda_ins.shape[0] and lambda_ins[pos] > self.min_action_score:
                    insert_pos = pos - 1
                    top_k_tokens = top_k if top_k else insert_probs.shape[2]
                    top_tokens = torch.topk(insert_probs[0, pos], min(top_k_tokens, insert_probs.shape[2]))

                    for token_idx, prob in zip(top_tokens.indices, top_tokens.values):
                        token_name = self.tokenizer.convert_ids_to_tokens([token_idx.item()])[0]

                        # 过滤无效的变量token
                        if valid_variables is not None:
                            # 检查是否是变量token（以'x'开头且后面跟数字）
                            if token_name.startswith('x') and token_name not in valid_variables:
                                continue  # 跳过无效的变量

                        new_tokens = current_tokens.copy()
                        new_tokens.insert(insert_pos, token_name)

                        proposals.append(ActionProposal(
                            action_type='insert',
                            position=insert_pos,
                            token=token_name,
                            score=lambda_ins[pos] * prob.item(),
                            new_tokens=new_tokens
                        ))

            # 生成substitute操作提案
            seq_len = min(effective_length, lambda_sub.shape[0])
            for pos in range(1, seq_len):
                current_token_idx = pos - 1
                if pos < lambda_sub.shape[0] and current_token_idx < len(current_tokens) and lambda_sub[pos] > self.min_action_score:
                    top_k_tokens = top_k if top_k else substitute_probs.shape[2]
                    top_tokens = torch.topk(substitute_probs[0, pos], min(top_k_tokens, substitute_probs.shape[2]))

                    for token_idx, prob in zip(top_tokens.indices, top_tokens.values):
                        token_name = self.tokenizer.convert_ids_to_tokens([token_idx.item()])[0]

                        # 过滤无效的变量token
                        if valid_variables is not None:
                            if token_name.startswith('x') and token_name not in valid_variables:
                                continue  # 跳过无效的变量

                        new_tokens = current_tokens.copy()
                        new_tokens[current_token_idx] = token_name

                        proposals.append(ActionProposal(
                            action_type='substitute',
                            position=current_token_idx,
                            token=token_name,
                            score=lambda_sub[pos] * prob.item(),
                            new_tokens=new_tokens
                        ))

            # 生成delete操作提案
            seq_len = min(effective_length, lambda_del.shape[0])
            for pos in range(1, seq_len):
                current_token_idx = pos - 1
                if pos < lambda_del.shape[0] and current_token_idx < len(current_tokens) and lambda_del[pos] > self.min_action_score:
                    if len(current_tokens) > 1:  # 保持至少一个token
                        new_tokens = current_tokens.copy()
                        del new_tokens[current_token_idx]

                        proposals.append(ActionProposal(
                            action_type='delete',
                            position=current_token_idx,
                            score=lambda_del[pos],
                            new_tokens=new_tokens
                        ))

        # 按分数排序
        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals

    def evaluate_candidate(self,
                          tokens: List[str],
                          x_data: np.ndarray,
                          y_data: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """评估候选表达式（支持常数优化）

        Args:
            tokens: token列表
            x_data: 输入x数据
            y_data: 目标y数据

        Returns:
            (成功标志, MSE分数, 残差)
        """
        try:
            from ..symbolic.symbolic_utils import (
                evaluate_expression_with_constants,
                evaluate_expression_safe,
                tree_to_expr
            )

            expr_str = ','.join(tokens)

            # 使用常数优化评估表达式（修复常数token不可导问题）
            success, optimized_expr, mse = evaluate_expression_with_constants(
                tree_str=expr_str,
                x_values=x_data,
                y_values=y_data
            )

            if success and optimized_expr is not None:
                # 用优化后的表达式重新计算预测值，获取residuals
                success2, y_pred = evaluate_expression_safe(optimized_expr, x_data)
                if success2:
                    y_pred_clipped = np.clip(y_pred, -self.numerical_clip_threshold, self.numerical_clip_threshold)
                    residuals = np.clip(y_data - y_pred_clipped, -self.numerical_clip_threshold, self.numerical_clip_threshold)
                    return True, -mse, residuals  # 负MSE作为分数（越高越好）
                else:
                    return False, float('-inf'), None
            else:
                return False, float('-inf'), None
        except Exception as e:
            return False, float('-inf'), None

    def beam_search(self,
                    initial_tokens: List[str],
                    initial_condition: torch.Tensor,
                    initial_residuals: np.ndarray,
                    x_data: np.ndarray,
                    y_data: np.ndarray,
                    x_values: torch.Tensor,
                    n_steps: int,
                    beam_size: int = 5,
                    top_k_actions: int = 10,
                    valid_variables: Optional[List[str]] = None) -> BeamCandidate:
        """执行束搜索

        Args:
            initial_tokens: 初始token列表
            initial_condition: 初始条件嵌入
            initial_residuals: 初始残差
            x_data: 输入x数据
            y_data: 目标y数据
            x_values: 输入x的tensor形式
            n_steps: 推理步数
            beam_size: 束大小
            top_k_actions: 每步考虑的top-k操作数
            valid_variables: 有效的变量token列表（如 ['x0', 'x1']），如果为None则从x_data推断

        Returns:
            最佳候选
        """
        # 如果没有提供有效变量列表，从x_data推断
        if valid_variables is None:
            input_dim = x_data.shape[1] if len(x_data.shape) > 1 else 1
            valid_variables = [f'x{i}' for i in range(input_dim)]

        # 记录有效变量信息（用于调试）
        if self.logger and self._is_main_process():
            self.logger.log("VALID_VARIABLES",
                           f"束搜索初始化 | input_dim={input_dim if len(x_data.shape) > 1 else 1} | "
                           f"valid_variables={valid_variables}",
                           "beam_search", level=1)

        # 初始化beam
        beam = [
            BeamCandidate(
                tokens=initial_tokens.copy(),
                score=0.0,
                condition=initial_condition,
                residuals=initial_residuals
            )
        ]

        for step in range(n_steps):
            if self.logger and self._is_main_process():
                print(f"\n束搜索步骤 {step + 1}/{n_steps}")
                print(f"当前beam大小: {len(beam)}")
                for i, cand in enumerate(beam[:3]):  # 显示前3个
                    print(f"  候选{i}: score={cand.score:.4f}, tokens={','.join(cand.tokens) if cand.tokens else '<blank>'}")

            # 所有候选的扩展
            all_candidates = []

            for candidate in beam:
                t = 0.1 + 0.9 * step / n_steps

                # 为当前候选生成操作提案
                proposals = self.generate_action_proposals(
                    current_tokens=candidate.tokens,
                    condition=candidate.condition,
                    x_values=x_values,
                    t=t,
                    top_k=top_k_actions,
                    valid_variables=valid_variables  # 传递有效变量列表
                )

                if not proposals:
                    # 没有有效操作，保留原候选
                    all_candidates.append(candidate)
                    continue

                # 只保留分数最高的N个操作
                top_proposals = proposals[:beam_size]

                for proposal in top_proposals:
                    # 评估新候选的表达式质量
                    success, expr_score, new_residuals = self.evaluate_candidate(
                        tokens=proposal.new_tokens,
                        x_data=x_data,
                        y_data=y_data
                    )

                    # 计算新分数：原分数 + 操作分数 + 表达式分数
                    # 这里使用加权和，可以调整权重
                    action_weight = 1.0
                    expr_weight = 0.1

                    new_score = candidate.score + action_weight * proposal.score
                    if success:
                        new_score += expr_weight * expr_score
                    else:
                        new_score -= 10.0  # 惩罚无效表达式

                    # 更新条件嵌入
                    if success and new_residuals is not None:
                        new_residuals_tensor = torch.FloatTensor(new_residuals).unsqueeze(0).to(self.device)
                        # 创建全1mask（因为这些都是真实点，没有padding）
                        new_point_mask = torch.ones_like(new_residuals_tensor)
                        new_condition = self.condition_encoder(
                            x_values,
                            new_residuals_tensor,
                            new_point_mask
                        )
                    else:
                        new_condition = candidate.condition

                    # 创建新候选
                    history = candidate.history.copy()
                    history.append(f"{proposal.action_type}@{proposal.position}:{proposal.token if proposal.token else 'N/A'}")

                    new_candidate = BeamCandidate(
                        tokens=proposal.new_tokens,
                        score=new_score,
                        condition=new_condition,
                        residuals=new_residuals if success else candidate.residuals,
                        history=history
                    )

                    all_candidates.append(new_candidate)

            # 如果没有任何有效扩展，提前终止
            if not all_candidates:
                break

            # 选择top-k候选
            all_candidates.sort(key=lambda c: c.score, reverse=True)
            beam = all_candidates[:beam_size]

        # 返回最佳候选
        best = beam[0] if beam else None

        if self.logger and best:
            self.logger.log("BEAM_SEARCH_COMPLETE",
                           f"最佳候选 | score={best.score:.4f} | tokens={','.join(best.tokens) if best.tokens else '<blank>'}",
                           "inference", level=1)

        return best

    def _is_main_process(self):
        """检查是否为主进程"""
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            return self.accelerator.is_local_main_process
        return True
