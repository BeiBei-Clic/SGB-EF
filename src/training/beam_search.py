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

            # 修复索引错位bug：模型输出顺序是 [ins_rate, del_rate, sub_rate]
            # 因此索引 0=插入, 1=删除, 2=替换
            lambda_ins = rates[0, :, 0].cpu().numpy()  # 插入速率
            lambda_del = rates[0, :, 1].cpu().numpy()  # 删除速率（修复：原来是 lambda_sub）
            lambda_sub = rates[0, :, 2].cpu().numpy()  # 替换速率（修复：原来是 lambda_del）

            # 调试：验证序列格式
            base_length = int(attention_mask[0].sum().item())
            effective_length = base_length  # 使用实际序列长度

            # 调试：打印序列格式信息（仅在第一次调用时）
            if not hasattr(self, '_sequence_format_logged'):
                if self.logger and self._is_main_process():
                    self.logger.log("SEQUENCE_FORMAT",
                                   f"推理序列格式验证 | input_ids[0:5]={input_ids[0, :5].tolist()} | "
                                   f"base_length={base_length} | effective_length={effective_length} | "
                                   f"current_tokens={current_tokens[:3]}",
                                   "beam_search", level=2)
                self._sequence_format_logged = True

            # 调试：打印top-5插入概率和token（仅在第一次调用时）
            if not hasattr(self, '_insert_probs_logged'):
                if self.logger and self._is_main_process():
                    for i in range(min(3, effective_length)):
                        top5 = torch.topk(insert_probs[0, i], 5)
                        tokens = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in top5.indices]
                        probs = top5.values.tolist()
                        self.logger.log("INSERT_PROBS_DEBUG",
                                       f"位置{i}: lambda={lambda_ins[i]:.4f} | top5_tokens={tokens} | top5_probs={probs}",
                                       "beam_search", level=2)
                self._insert_probs_logged = True

            # 生成insert操作提案
            # 关键修复：训练-推理索引对齐
            # 训练时(fill_gap_tokens_with_repeats)：
            #   - z_t = [<s>, <gap>, add, ...]
            #   - <gap>位置使用前一个非gap位置的预测
            #   - 在最开头的<gap>使用位置0(<s>)的预测
            # 推理时应该保持一致：
            #   - 在最开头插入时，使用BOS的预测
            #   - 在其他位置插入时，使用前一个token的预测
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
                    insert_pos = current_token_pos  # 在current_tokens[current_token_pos]之后插入
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
                    valid_variables: Optional[List[str]] = None,
                    return_all_candidates: bool = False) -> Tuple[BeamCandidate, Optional[List[BeamCandidate]]]:
        """执行束搜索

        束搜索策略：使用模型预测的操作分数作为引导，扩大搜索空间，
        最后从所有候选中根据MSE选择最佳表达式。

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
            return_all_candidates: 是否返回所有有效候选

        Returns:
            (MSE最佳候选, 所有有效候选列表)
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

        # 初始化beam - 使用操作分数作为引导（不使用MSE）
        # 初始分数设为0，表示探索起点
        beam = [
            BeamCandidate(
                tokens=initial_tokens.copy(),
                score=0.0,  # 初始分数，仅用于操作引导
                condition=initial_condition,
                residuals=initial_residuals
            )
        ]

        # 记录历史表达式，避免重复探索
        seen_expressions = {tuple(initial_tokens)}

        # 收集所有有效候选（用于最终MSE选择）
        all_valid_candidates = []

        for step in range(n_steps):
            if self.logger and self._is_main_process():
                print(f"\n束搜索步骤 {step + 1}/{n_steps}")
                print(f"当前beam大小: {len(beam)}")
                for i, cand in enumerate(beam[:3]):  # 显示前3个
                    print(f"  候选{i}: score={cand.score:.4f}, tokens={','.join(cand.tokens) if cand.tokens else '<blank>'}")

            # 使用字典去重：相同表达式只保留操作分数最高的
            candidate_map = {}

            for candidate in beam:
                t = 0.1 + 0.9 * step / n_steps

                # 为当前候选生成操作提案
                proposals = self.generate_action_proposals(
                    current_tokens=candidate.tokens,
                    condition=candidate.condition,
                    x_values=x_values,
                    t=t,
                    top_k=top_k_actions,
                    valid_variables=valid_variables
                )

                if not proposals:
                    # 没有有效操作，保留原候选
                    cand_key = tuple(candidate.tokens)
                    if cand_key not in candidate_map or candidate.score > candidate_map[cand_key].score:
                        candidate_map[cand_key] = candidate
                    continue

                # 只保留分数最高的N个操作
                top_proposals = proposals[:top_k_actions]

                for proposal in top_proposals:
                    # 跳过已经见过的表达式
                    prop_key = tuple(proposal.new_tokens)
                    if prop_key in seen_expressions:
                        continue

                    # 跳过过长的表达式
                    if len(proposal.new_tokens) > self.max_expression_length:
                        continue

                    # 新分数使用操作分数累积（模型预测的引导）
                    # 不在这里使用MSE，仅用模型预测的操作分数来引导搜索
                    new_score = candidate.score + proposal.score

                    # 添加长度惩罚：偏好更简洁的表达式
                    length_penalty = 0.001 * len(proposal.new_tokens)
                    new_score -= length_penalty

                    # 临时保存候选，稍后统一评估MSE
                    new_candidate = BeamCandidate(
                        tokens=proposal.new_tokens,
                        score=new_score,  # 操作分数
                        condition=candidate.condition,  # 暂时沿用原条件
                        residuals=candidate.residuals,
                        history=candidate.history + [f"{proposal.action_type}@{proposal.position}:{proposal.token if proposal.token else 'N/A'}(score={proposal.score:.4f})"]
                    )

                    # 去重：相同表达式只保留操作分数最高的
                    if prop_key not in candidate_map or new_score > candidate_map[prop_key].score:
                        candidate_map[prop_key] = new_candidate
                        seen_expressions.add(prop_key)

            # 如果没有任何有效扩展，提前终止
            if not candidate_map:
                break

            # 转换为列表并按操作分数排序
            all_candidates = list(candidate_map.values())
            all_candidates.sort(key=lambda c: c.score, reverse=True)
            beam = all_candidates[:beam_size]

            # 检查是否收敛（操作分数连续多步无改善）
            if step > 5 and len(beam) > 0:
                best_current_score = beam[0].score
                if step > 0 and hasattr(self, '_prev_best_score'):
                    if abs(best_current_score - self._prev_best_score) < 1e-6:
                        self._convergence_count = getattr(self, '_convergence_count', 0) + 1
                        if self._convergence_count >= 3:
                            if self.logger and self._is_main_process():
                                print(f"\n收敛检测：操作分数连续3步无改善，提前终止")
                            break
                    else:
                        self._convergence_count = 0
                self._prev_best_score = best_current_score

        # 收集所有有效候选并评估MSE
        # 将初始表达式和所有探索到的表达式都加入候选池
        exploration_pool = list(candidate_map.values()) if candidate_map else []
        exploration_pool.extend(beam)  # 确保当前beam也被包含

        for candidate in exploration_pool:
            # 评估每个候选的MSE
            success, expr_mse_score, _ = self.evaluate_candidate(
                tokens=candidate.tokens,
                x_data=x_data,
                y_data=y_data
            )
            if success:
                # 保存MSE分数到候选对象
                candidate.mse_score = -expr_mse_score  # 转换回正MSE
                all_valid_candidates.append(candidate)

        # 如果没有任何有效候选，返回初始表达式
        if not all_valid_candidates:
            # 至少评估初始表达式
            init_success, init_mse_score, _ = self.evaluate_candidate(
                tokens=initial_tokens,
                x_data=x_data,
                y_data=y_data
            )
            best_candidate = BeamCandidate(
                tokens=initial_tokens.copy(),
                score=0.0,
                condition=initial_condition,
                residuals=initial_residuals
            )
            if init_success:
                best_candidate.mse_score = -init_mse_score
            return best_candidate, None

        # 按MSE排序，选择最佳
        all_valid_candidates.sort(key=lambda c: c.mse_score)
        best_candidate = all_valid_candidates[0]

        if self.logger and self._is_main_process():
            self.logger.log("BEAM_SEARCH_COMPLETE",
                           f"MSE最佳候选 | MSE={best_candidate.mse_score:.6f} | tokens={','.join(best_candidate.tokens) if best_candidate.tokens else '<blank>'} | "
                           f"探索了{len(all_valid_candidates)}个有效候选",
                           "inference", level=1)
            print(f"\n束搜索完成:")
            print(f"  探索了 {len(all_valid_candidates)} 个有效表达式")
            print(f"  最佳MSE: {best_candidate.mse_score:.6f}")
            print(f"  最佳表达式: {','.join(best_candidate.tokens) if best_candidate.tokens else '<blank>'}")
            # 显示top-5
            for i, cand in enumerate(all_valid_candidates[:5]):
                print(f"  Top-{i+1}: MSE={cand.mse_score:.6f}, expr={','.join(cand.tokens) if cand.tokens else '<blank>'}")

        return best_candidate, (all_valid_candidates if return_all_candidates else None)

    def _is_main_process(self):
        """检查是否为主进程"""
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            return self.accelerator.is_local_main_process
        return True
