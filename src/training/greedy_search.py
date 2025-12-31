"""
简单推理模块 - 用于符号回归推理的贪婪搜索算法

架构说明（v2.0 - 迭代优化模式）:
- 推理时固定 t=0，与训练时完全一致
- 条件编码器使用目标值y_target（北极星模式）
- 每步都从"当前状态"预测"如何编辑到目标状态"
- 移除了旧架构的"渐进式时间步"逻辑
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(order=True)
class Candidate:
    """候选表达式

    Attributes:
        score: 累积操作分数（越高越好，用于搜索引导）
        tokens: 表达式token列表
        condition: 对应的条件嵌入
        residuals: 对应的残差
        history: 操作历史（可选，用于调试）
        mse_score: 表达式的MSE分数（越低越好，用于最终选择）
    """
    score: float
    tokens: List[str] = field(compare=False)
    condition: Optional[torch.Tensor] = field(default=None, compare=False)
    residuals: Optional[np.ndarray] = field(default=None, compare=False)
    history: List[str] = field(default_factory=list, compare=False)
    mse_score: Optional[float] = field(default=None, compare=False)

    def __repr__(self):
        return f"Candidate(score={self.score:.4f}, tokens={','.join(self.tokens) if self.tokens else '<blank>'})"


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


class SimpleSymbolicRegression:
    """简单符号回归推理器 - 使用贪婪搜索(每次选择最佳操作)

    架构v2.0更新：
    - 推理时固定 t=0（与训练一致），不再使用渐进式时间步
    - 条件嵌入使用目标值y_target，在整个推理过程中保持恒定
    - 模型学习"从当前状态到目标状态的直接编辑"
    """

    def __init__(self,
                 model,
                 condition_encoder,
                 tokenizer,
                 # special_tokens_manager,  # 已移除：使用小词表后不再需要
                 device,
                 args,
                 logger,
                 min_action_score=0.01,
                 max_expression_length=50,
                 numerical_clip_threshold=1e6):
        """初始化简单推理器

        Args:
            model: EditFlow模型
            condition_encoder: 条件编码器
            tokenizer: 分词器
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
        # self.special_tokens_manager = special_tokens_manager  # 已移除：使用小词表后不再需要
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
                                  top_k: Optional[int] = None,
                                  valid_variables: Optional[List[str]] = None,
                                  step: int = -1) -> List[ActionProposal]:
        """为当前表达式生成操作提案

        Args:
            current_tokens: 当前token列表
            condition: 条件嵌入
            x_values: 输入x值
            top_k: 每种操作类型保留的top-k数量，None表示全部
            valid_variables: 有效的变量token列表（如 ['x0', 'x1']），用于过滤无效变量
            step: 当前推理步数（用于日志记录）

        Returns:
            操作提案列表
        """
        import time
        proposals = []

        # 记录开始时间用于性能统计
        start_time = time.time()

        # 构建模型输入 - 确保与训练时的序列格式完全一致
        tokenized_expr = self.tokenizer.convert_tokens_to_ids(current_tokens)
        max_len = getattr(self.args, 'max_expr_length', 128)

        if len(tokenized_expr) > max_len - 1:
            tokenized_expr = tokenized_expr[:max_len-1]

        bos_token = self.tokenizer.convert_tokens_to_ids('<s>')
        pad_token = self.tokenizer.convert_tokens_to_ids('<pad>')

        # 关键：必须添加 BOS token，与训练时的格式一致 [BOS] + tokens + [PAD]
        tokenized_expr = [bos_token] + tokenized_expr
        tokenized_expr = tokenized_expr + [pad_token] * (max_len - len(tokenized_expr))

        # 验证 BOS token 在正确位置
        assert tokenized_expr[0] == bos_token, f"BOS token 必须在位置0，但得到 {tokenized_expr[0]}"

        input_ids = torch.LongTensor([tokenized_expr]).to(self.device)
        attention_mask = (input_ids != pad_token).float().to(self.device)

        with torch.no_grad():
            model_forward_start = time.time()
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                condition=condition
            )
            model_forward_time = (time.time() - model_forward_start) * 1000  # ms

            # 从字典中提取结果
            ins_rate, del_rate, sub_rate = output['rates']
            rates = torch.cat([ins_rate, del_rate, sub_rate], dim=-1)
            insert_probs = output['insert_probs']
            substitute_probs = output['substitute_probs']

            # 修复索引错位bug：模型输出顺序是 [ins_rate, del_rate, sub_rate]
            # 因此索引 0=插入, 1=删除, 2=替换
            lambda_ins = rates[0, :, 0].cpu().numpy()  # 插入速率
            lambda_del = rates[0, :, 1].cpu().numpy()  # 删除速率
            lambda_sub = rates[0, :, 2].cpu().numpy()  # 替换速率

            # 统计操作速率信息
            ins_rate_mean = float(lambda_ins.mean())
            ins_rate_max = float(lambda_ins.max())
            ins_rate_above_threshold = int(np.sum(lambda_ins > self.min_action_score))

            del_rate_mean = float(lambda_del.mean())
            del_rate_max = float(lambda_del.max())
            del_rate_above_threshold = int(np.sum(lambda_del > self.min_action_score))

            sub_rate_mean = float(lambda_sub.mean())
            sub_rate_max = float(lambda_sub.max())
            sub_rate_above_threshold = int(np.sum(lambda_sub > self.min_action_score))

            # 调试：验证序列格式
            base_length = int(attention_mask[0].sum().item())
            effective_length = base_length  # 使用实际序列长度

            # 调试：打印序列格式信息（每次调用都打印）
            if self.logger and self._is_main_process():
                self.logger.log_greedy_search_sequence_format(
                    input_ids[0, :5].tolist(),
                    base_length,
                    effective_length,
                    current_tokens[:3],
                    level=3
                )

            # 调试：打印top-10插入概率和token
            if self.logger and self._is_main_process():
                self.logger.log_greedy_search_separator("插入操作详细预测信息（前3个位置）", level=3)
                for i in range(min(3, effective_length)):
                    top10 = torch.topk(insert_probs[0, i], 10)
                    tokens = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in top10.indices]
                    probs = top10.values.tolist()
                    self.logger.log_greedy_search_insert_probs(i, lambda_ins[i], tokens, probs, level=3)
                self.logger.log_greedy_search_separator(level=3)

            seq_len = min(effective_length, lambda_ins.shape[0])

            # 特殊情况：在最开头插入时，使用BOS的预测（位置0）
            if 0 < lambda_ins.shape[0] and lambda_ins[0] > self.min_action_score:
                insert_pos = 0  # 在current_tokens最前面插入
                top_k_tokens = top_k if top_k else insert_probs.shape[2]
                # 使用位置0（BOS）的预测
                top_tokens = torch.topk(insert_probs[0, 0], min(top_k_tokens, insert_probs.shape[2]))

                for token_idx, prob in zip(top_tokens.indices, top_tokens.values):
                    token_name = self.tokenizer.convert_ids_to_tokens([token_idx.item()])[0]

                    if valid_variables is not None:
                        if token_name.startswith('x') and token_name not in valid_variables:
                            continue

                    new_tokens = current_tokens.copy()
                    new_tokens.insert(insert_pos, token_name)

                    proposals.append(ActionProposal(
                        action_type='insert',
                        position=insert_pos,
                        token=token_name,
                        score=lambda_ins[0] * prob.item(),
                        new_tokens=new_tokens
                    ))

            # 遍历每个token位置（在current_tokens中的索引）
            # 在位置p之后插入，使用input_ids[p+1]的预测（因为input_ids[0]是BOS）
            for current_token_pos in range(len(current_tokens)):
                input_ids_pos = current_token_pos + 1  # +1因为input_ids[0]是BOS
                if input_ids_pos < lambda_ins.shape[0] and lambda_ins[input_ids_pos] > self.min_action_score:
                    # 关键修复：Python的list.insert(i, x)是在索引i之前插入
                    # 要在current_tokens[current_token_pos]之后插入，需要用current_token_pos + 1
                    insert_pos = current_token_pos + 1
                    top_k_tokens = top_k if top_k else insert_probs.shape[2]
                    # 使用input_ids[input_ids_pos]的预测
                    top_tokens = torch.topk(insert_probs[0, input_ids_pos], min(top_k_tokens, insert_probs.shape[2]))

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
                            score=lambda_ins[input_ids_pos] * prob.item(),
                            new_tokens=new_tokens
                        ))

            # 生成substitute操作提案
            seq_len = min(effective_length, lambda_sub.shape[0])

            # 调试：打印top-10替换概率和token
            if self.logger and self._is_main_process():
                self.logger.log_greedy_search_separator("替换操作详细预测信息（前3个token位置）", level=3)
                for idx in range(min(3, len(current_tokens))):
                    pos = idx + 1  # +1因为input_ids[0]是BOS
                    if pos < substitute_probs.shape[1]:
                        top10 = torch.topk(substitute_probs[0, pos], 10)
                        tokens = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in top10.indices]
                        probs = top10.values.tolist()
                        current_token = current_tokens[idx] if idx < len(current_tokens) else 'N/A'
                        self.logger.log_greedy_search_substitute_probs(idx, current_token, lambda_sub[pos], tokens, probs, level=3)
                self.logger.log_greedy_search_separator(level=3)

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

            # 调试：打印删除速率信息（每次调用都打印）
            if self.logger and self._is_main_process():
                self.logger.log_greedy_search_separator("删除操作详细预测信息（所有token位置）", level=3)
                for idx in range(len(current_tokens)):
                    pos = idx + 1  # +1因为input_ids[0]是BOS
                    if pos < lambda_del.shape[0]:
                        current_token = current_tokens[idx] if idx < len(current_tokens) else 'N/A'
                        self.logger.log_greedy_search_delete_probs(idx, current_token, lambda_del[pos],
                                                                lambda_del[pos] > self.min_action_score, level=3)
                self.logger.log_greedy_search_separator(level=3)

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

        # 记录提案生成性能和统计信息
        total_time = (time.time() - start_time) * 1000  # ms

        # 统计各类操作数量
        insert_count = sum(1 for p in proposals if p.action_type == 'insert')
        delete_count = sum(1 for p in proposals if p.action_type == 'delete')
        substitute_count = sum(1 for p in proposals if p.action_type == 'substitute')

        if self.logger and self._is_main_process():
            self.logger.log("ACTION_PROPOSALS_GENERATED",
                           f"step={step} | "
                           f"proposals={len(proposals)} (ins={insert_count}, del={delete_count}, sub={substitute_count}) | "
                           f"rates: ins(mean={ins_rate_mean:.4f}, max={ins_rate_max:.4f}, >thr={ins_rate_above_threshold}) "
                           f"del(mean={del_rate_mean:.4f}, max={del_rate_max:.4f}, >thr={del_rate_above_threshold}) "
                           f"sub(mean={sub_rate_mean:.4f}, max={sub_rate_max:.4f}, >thr={sub_rate_above_threshold}) | "
                           f"time: model={model_forward_time:.1f}ms total={total_time:.1f}ms | "
                           f"current_tokens={','.join(current_tokens) if len(current_tokens)<=10 else ','.join(current_tokens[:10])+'...'}",
                           "simple_search", level=3)

        return proposals

    def evaluate_candidate(self,
                          tokens: List[str],
                          x_data: np.ndarray,
                          y_data: np.ndarray,
                          step: int = -1,
                          candidate_id: str = "") -> Tuple[bool, float, np.ndarray]:
        """评估候选表达式（支持常数优化）

        Args:
            tokens: token列表
            x_data: 输入x数据
            y_data: 目标y数据
            step: 当前推理步数（用于日志记录）
            candidate_id: 候选标识（用于日志记录）

        Returns:
            (成功标志, MSE分数, 残差)
        """
        import time
        try:
            eval_start = time.time()

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

            eval_time = (time.time() - eval_start) * 1000  # ms

            # 记录表达式评估性能
            if self.logger and self._is_main_process():
                expr_short = expr_str if len(expr_str) <= 30 else expr_str[:30] + "..."
                self.logger.log("EXPR_EVALUATION",
                               f"step={step} | candidate={candidate_id} | expr='{expr_short}' | "
                               f"success={success} | mse={mse:.6f if success else 'N/A'} | "
                               f"optimized_expr={optimized_expr if optimized_expr else 'N/A'} | "
                               f"time={eval_time:.1f}ms",
                               "simple_search", level=3)

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

    def greedy_search(self,
                      initial_tokens: List[str],
                      initial_condition: torch.Tensor,
                      initial_residuals: np.ndarray,
                      x_data: np.ndarray,
                      y_data: np.ndarray,
                      x_values: torch.Tensor,
                      n_steps: int,
                      valid_variables: Optional[List[str]] = None) -> Candidate:
        """执行贪婪搜索推理（架构v2.0 - 固定t=0模式）

        每一步选择模型预测的最佳操作并执行,不维护多个候选

        架构关键改进：
        - 固定 t=0：与训练时完全一致，避免分布漂移
        - 恒定条件：initial_condition在推理过程中不变（北极星模式）
        - 简化逻辑：不再需要渐进式时间步调度

        Args:
            initial_tokens: 初始token列表
            initial_condition: 初始条件嵌入（使用y_target生成，恒定不变）
            initial_residuals: 初始残差（仅用于日志，不作为条件）
            x_data: 输入x数据
            y_data: 目标y数据
            x_values: 输入x的tensor形式
            n_steps: 推理步数
            valid_variables: 有效的变量token列表（如 ['x0', 'x1']），如果为None则从x_data推断

        Returns:
            最佳候选
        """
        # 如果没有提供有效变量列表，从x_data推断
        if valid_variables is None:
            input_dim = x_data.shape[1] if len(x_data.shape) > 1 else 1
            valid_variables = [f'x{i}' for i in range(input_dim)]

        # 记录有效变量信息
        if self.logger and self._is_main_process():
            self.logger.log("VALID_VARIABLES",
                           f"贪婪搜索初始化 | input_dim={input_dim if len(x_data.shape) > 1 else 1} | "
                           f"valid_variables={valid_variables}",
                           "greedy_search", level=3)
            print(f"\n开始贪婪搜索推理 (共{n_steps}步)")
            print(f"初始表达式: {','.join(initial_tokens)}")

        # 初始化当前候选
        current_candidate = Candidate(
            tokens=initial_tokens.copy(),
            score=0.0,
            condition=initial_condition,
            residuals=initial_residuals
        )

        # 记录操作历史
        history = []

        for step in range(n_steps):
            if self.logger and self._is_main_process():
                print(f"\n步骤 {step + 1}/{n_steps}")
                print(f"当前表达式: {','.join(current_candidate.tokens)}")

                # 记录残差信息
                if current_candidate.residuals is not None:
                    residuals = current_candidate.residuals
                    self.logger.log("STEP_RESIDUALS",
                                   f"step={step} | "
                                   f"residuals: mean={residuals.mean():.6f}, std={residuals.std():.6f}, "
                                   f"min={residuals.min():.6f}, max={residuals.max():.6f}, "
                                   f"l2_norm={np.linalg.norm(residuals):.6f}",
                                   "greedy_search", level=3)

                # 记录条件嵌入信息
                if current_candidate.condition is not None:
                    condition = current_candidate.condition
                    if condition.dim() == 3:
                        cond_flat = condition.detach().cpu().flatten().numpy()
                    else:
                        cond_flat = condition.detach().cpu().squeeze(0).numpy()
                    self.logger.log("STEP_CONDITION",
                                   f"step={step} | "
                                   f"condition: shape={list(condition.shape)}, "
                                   f"mean={cond_flat.mean():.6f}, std={cond_flat.std():.6f}, "
                                   f"min={cond_flat.min():.6f}, max={cond_flat.max():.6f}",
                                   "greedy_search", level=3)

            # 为当前候选生成操作提案
            proposals = self.generate_action_proposals(
                current_tokens=current_candidate.tokens,
                condition=current_candidate.condition,
                x_values=x_values,
                top_k=None,  # 获取所有操作提案
                valid_variables=valid_variables,
                step=step
            )

            if not proposals:
                if self.logger and self._is_main_process():
                    print(f"没有找到有效操作,提前终止")
                    self.logger.log("GREEDY_SEARCH_STOP",
                                   f"step={step} | 没有找到有效操作,提前终止",
                                   "greedy_search", level=2)
                break

            # 选择分数最高的操作
            best_proposal = proposals[0]
            action_desc = f"{best_proposal.action_type}@{best_proposal.position}"
            if best_proposal.token:
                action_desc += f":{best_proposal.token}"
            action_desc += f"(score={best_proposal.score:.4f})"

            history.append(action_desc)

            if self.logger and self._is_main_process():
                print(f"执行操作: {action_desc}")
                self.logger.log("GREEDY_SEARCH_ACTION",
                               f"step={step} | action={action_desc} | "
                               f"old_expr={','.join(current_candidate.tokens)} | "
                               f"new_expr={','.join(best_proposal.new_tokens)}",
                               "greedy_search", level=2)

            # 更新当前候选
            # 注意：不重新计算条件嵌入（架构v2.0：条件恒定）
            current_candidate = Candidate(
                tokens=best_proposal.new_tokens,
                score=current_candidate.score + best_proposal.score,
                condition=current_candidate.condition,  # 条件保持恒定
                residuals=current_candidate.residuals,   # 残差仅用于日志
                history=history.copy()
            )

            # 架构v2.0：不重新计算条件嵌入（北极星模式）
            # 旧架构：每步重新计算残差和条件（导致分布漂移）
            # 新架构：条件恒定，避免分布漂移，提供稳定的优化方向

        # 评估最终表达式的MSE
        if self.logger and self._is_main_process():
            print(f"\n贪婪搜索完成")
            print(f"最终表达式: {','.join(current_candidate.tokens)}")

        # 评估MSE
        success, expr_mse_score, _ = self.evaluate_candidate(
            tokens=current_candidate.tokens,
            x_data=x_data,
            y_data=y_data,
            step=n_steps,
            candidate_id="final"
        )

        if success:
            current_candidate.mse_score = -expr_mse_score  # 转换为正MSE
            if self.logger and self._is_main_process():
                print(f"最终MSE: {current_candidate.mse_score:.6f}")
                self.logger.log("GREEDY_SEARCH_COMPLETE",
                               f"tokens={','.join(current_candidate.tokens)} | "
                               f"MSE={current_candidate.mse_score:.6f} | "
                               f"总操作数={len(history)}",
                               "inference", level=1)
        else:
            if self.logger and self._is_main_process():
                print(f"无法评估最终表达式的MSE")
                self.logger.log("GREEDY_SEARCH_FAILED",
                               f"无法评估最终表达式 | tokens={','.join(current_candidate.tokens)}",
                               "inference", level=1)

        return current_candidate

    def _is_main_process(self):
        """检查是否为主进程"""
        return True
