"""
简单推理模块 - 用于符号回归推理的贪婪搜索算法

架构说明（v2.0 - 迭代优化模式）:
- 推理时固定 t=0，与训练时完全一致
- 条件编码器使用目标值y_target（北极星模式）
- 每步都从"当前状态"预测"如何编辑到目标状态"
- 移除了旧架构的"渐进式时间步"逻辑
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============= 数据类定义 =============
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
        position_actions_history: 每步每个位置的操作详情 [{position: (action_type, score)}, ...]
    """
    score: float
    tokens: List[str] = field(compare=False)
    condition: Optional[torch.Tensor] = field(default=None, compare=False)
    residuals: Optional[np.ndarray] = field(default=None, compare=False)
    history: List[str] = field(default_factory=list, compare=False)
    mse_score: Optional[float] = field(default=None, compare=False)
    position_actions_history: List[dict] = field(default_factory=list, compare=False)

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
        elif self.action_type == 'keep':
            return f"Keep(pos={self.position}, score={self.score:.4f})"
        return f"Action({self.action_type}, score={self.score:.4f})"


# ============= 推理器类 =============
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
            min_action_score: 最小操作分数阈值（用于过滤低分操作）
            max_expression_length: 表达式最大长度
            numerical_clip_threshold: 数值裁剪阈值
        """
        self.model = model
        self.condition_encoder = condition_encoder
        self.tokenizer = tokenizer
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
        """为当前表达式生成操作提案"""
        proposals = []

        # 构建模型输入
        tokenized_expr = self.tokenizer.convert_tokens_to_ids(current_tokens)
        max_len = getattr(self.args, 'max_expr_length', 128)

        if len(tokenized_expr) > max_len - 1:
            tokenized_expr = tokenized_expr[:max_len-1]

        bos_token = self.tokenizer.convert_tokens_to_ids('<s>')
        pad_token = self.tokenizer.convert_tokens_to_ids('<pad>')

        # 必须添加 BOS token，与训练时的格式一致 [BOS] + tokens + [PAD]
        tokenized_expr = [bos_token] + tokenized_expr
        tokenized_expr = tokenized_expr + [pad_token] * (max_len - len(tokenized_expr))

        input_ids = torch.LongTensor([tokenized_expr]).to(self.device)
        attention_mask = (input_ids != pad_token).float().to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                condition=condition
            )

            # 提取结果
            rates_logits = output['rates_logits']
            rates_probs = F.softmax(rates_logits, dim=-1)

            lambda_ins = rates_probs[0, :, 0].cpu().numpy()  # 插入概率
            lambda_del = rates_probs[0, :, 1].cpu().numpy()  # 删除概率
            lambda_sub = rates_probs[0, :, 2].cpu().numpy()  # 替换概率
            lambda_keep = rates_probs[0, :, 3].cpu().numpy()  # 保持概率

            insert_probs = output['insert_probs']
            substitute_probs = output['substitute_probs']

            effective_length = int(attention_mask[0].sum().item())
            seq_len = min(effective_length, lambda_ins.shape[0])

            # 生成INSERT操作提案
            if 0 < lambda_ins.shape[0] and lambda_ins[0] > self.min_action_score:
                top_k_tokens = top_k if top_k else insert_probs.shape[2]
                top_tokens = torch.topk(insert_probs[0, 0], min(top_k_tokens, insert_probs.shape[2]))

                for token_idx, prob in zip(top_tokens.indices, top_tokens.values):
                    token_name = self.tokenizer.convert_ids_to_tokens([token_idx.item()])[0]

                    if valid_variables is not None:
                        if token_name.startswith('x') and token_name not in valid_variables:
                            continue

                    new_tokens = current_tokens.copy()
                    new_tokens.insert(0, token_name)

                    proposals.append(ActionProposal(
                        action_type='insert',
                        position=0,
                        token=token_name,
                        score=lambda_ins[0] * prob.item(),
                        new_tokens=new_tokens
                    ))

            # 遍历每个token位置生成INSERT
            for current_token_pos in range(len(current_tokens)):
                input_ids_pos = current_token_pos + 1
                if input_ids_pos < lambda_ins.shape[0] and lambda_ins[input_ids_pos] > self.min_action_score:
                    insert_pos = current_token_pos + 1
                    top_k_tokens = top_k if top_k else insert_probs.shape[2]
                    top_tokens = torch.topk(insert_probs[0, input_ids_pos], min(top_k_tokens, insert_probs.shape[2]))

                    for token_idx, prob in zip(top_tokens.indices, top_tokens.values):
                        token_name = self.tokenizer.convert_ids_to_tokens([token_idx.item()])[0]

                        if valid_variables is not None and token_name.startswith('x') and token_name not in valid_variables:
                            continue

                        new_tokens = current_tokens.copy()
                        new_tokens.insert(insert_pos, token_name)

                        proposals.append(ActionProposal(
                            action_type='insert',
                            position=insert_pos,
                            token=token_name,
                            score=lambda_ins[input_ids_pos] * prob.item(),
                            new_tokens=new_tokens
                        ))

            # 生成SUBSTITUTE操作提案
            seq_len = min(effective_length, lambda_sub.shape[0])
            for pos in range(1, seq_len):
                current_token_idx = pos - 1
                if pos < lambda_sub.shape[0] and current_token_idx < len(current_tokens) and lambda_sub[pos] > self.min_action_score:
                    top_k_tokens = top_k if top_k else substitute_probs.shape[2]
                    top_tokens = torch.topk(substitute_probs[0, pos], min(top_k_tokens, substitute_probs.shape[2]))

                    for token_idx, prob in zip(top_tokens.indices, top_tokens.values):
                        token_name = self.tokenizer.convert_ids_to_tokens([token_idx.item()])[0]

                        if valid_variables is not None and token_name.startswith('x') and token_name not in valid_variables:
                            continue

                        new_tokens = current_tokens.copy()
                        new_tokens[current_token_idx] = token_name

                        proposals.append(ActionProposal(
                            action_type='substitute',
                            position=pos,
                            token=token_name,
                            score=lambda_sub[pos] * prob.item(),
                            new_tokens=new_tokens
                        ))

            # 生成DELETE操作提案
            seq_len = min(effective_length, lambda_del.shape[0])
            for pos in range(1, seq_len):
                current_token_idx = pos - 1
                if pos < lambda_del.shape[0] and current_token_idx < len(current_tokens) and lambda_del[pos] > self.min_action_score:
                    if len(current_tokens) > 1:
                        new_tokens = current_tokens.copy()
                        del new_tokens[current_token_idx]

                        proposals.append(ActionProposal(
                            action_type='delete',
                            position=pos,
                            score=lambda_del[pos],
                            new_tokens=new_tokens
                        ))

            # 生成KEEP操作提案
            seq_len = min(effective_length, lambda_keep.shape[0])
            for pos in range(1, seq_len):
                current_token_idx = pos - 1
                if pos < lambda_keep.shape[0] and current_token_idx < len(current_tokens) and lambda_keep[pos] > self.min_action_score:
                    proposals.append(ActionProposal(
                        action_type='keep',
                        position=pos,
                        score=lambda_keep[pos],
                        new_tokens=current_tokens.copy()
                    ))

        # 按分数排序
        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals

    def apply_multiple_actions(self,
                               current_tokens: List[str],
                               actions: List[ActionProposal]) -> List[str]:
        """同时应用多个操作到表达式，智能处理操作间的位置依赖关系

        核心思路：
        1. 操作按原始位置从前往后应用
        2. 每次操作后，动态更新后续操作的位置
        3. 插入操作使后续位置+1，删除操作使后续位置-1

        Args:
            current_tokens: 当前token列表
            actions: 要应用的操作列表（应该已经按阈值过滤）

        Returns:
            应用所有操作后的新token列表
        """
        if not actions:
            return current_tokens.copy()

        if len(actions) == 1:
            # 单个操作，直接返回
            return actions[0].new_tokens.copy()

        # 多个操作：需要智能处理位置依赖
        # 策略：从前往后应用，每次操作后更新后续操作的位置

        # 1. 按位置升序排序（从前往后应用）
        sorted_actions = sorted(actions, key=lambda a: a.position)

        # 2. 初始化：从当前表达式开始
        result_tokens = current_tokens.copy()

        # 3. 用于追踪位置偏移量
        # 偏移量 = 已应用的插入数 - 已应用的删除数
        position_offset = 0

        # 4. 从前往后应用操作
        for action in sorted_actions:
            # 计算实际位置：将input_ids位置转换为current_tokens位置
            # input_ids: [BOS, token1, token2, ...]
            # current_tokens: [token1, token2, ...]
            if action.action_type == 'insert':
                # INSERT操作：position表示在哪个input_ids位置之后插入
                # INSERT(0)在BOS之后插入 → current_tokens索引0
                # INSERT(p)在位置p之后插入 → current_tokens索引p（如果p=0）或p-1之后（如果p>0）
                if action.position == 0:
                    # 在BOS之后插入，插入到current_tokens开头
                    actual_position = 0 + position_offset
                else:
                    # 在位置p之后插入，插入到current_tokens的位置p（因为current_tokens不包含BOS）
                    actual_position = action.position + position_offset
            else:
                # DELETE/SUBSTITUTE/KEEP操作：position指向input_ids位置
                # 需要转换为current_tokens位置：pos-1（跳过BOS）
                actual_position = (action.position - 1) + position_offset

            # 确保位置在有效范围内
            actual_position = max(0, min(actual_position, len(result_tokens)))

            if action.action_type == 'delete':
                # 删除操作
                if 0 <= actual_position < len(result_tokens) and len(result_tokens) > 1:
                    deleted_token = result_tokens[actual_position]
                    del result_tokens[actual_position]
                    position_offset -= 1  # 后续位置前移
                    # Debug: 只在main process打印
                    if os.environ.get('RANK', '0') == '0':
                        print(f"    [DEBUG] Delete @{action.position}→actual@{actual_position}: '{deleted_token}', 剩余{len(result_tokens)} tokens, offset={position_offset}")

            elif action.action_type == 'insert':
                # 插入操作
                if action.token is not None:
                    # 检查当前位置是否是<gap>
                    if (0 <= actual_position < len(result_tokens) and
                        result_tokens[actual_position] == '<gap>'):
                        # 如果是gap，直接替换（gap只是占位符，插入会消耗gap）
                        result_tokens[actual_position] = action.token
                    else:
                        # 如果不是gap，在actual_position之前插入
                        result_tokens.insert(actual_position, action.token)
                        position_offset += 1  # 后续位置后移

            elif action.action_type == 'substitute':
                # 替换操作：不影响位置偏移
                if 0 <= actual_position < len(result_tokens) and action.token is not None:
                    result_tokens[actual_position] = action.token

            elif action.action_type == 'keep':
                # KEEP操作：保持当前token不变，不执行任何修改
                pass

        return result_tokens

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
        try:
            eval_start = time.time()

            from ..symbolic.symbolic_utils import (
                evaluate_expression_with_constants,
                evaluate_expression_safe
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
        except Exception:
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

        if self._is_main_process():
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
            if self._is_main_process():
                print(f"步骤 {step + 1}/{n_steps}: {','.join(current_candidate.tokens)}")

            # 生成操作提案
            proposals = self.generate_action_proposals(
                current_tokens=current_candidate.tokens,
                condition=current_candidate.condition,
                x_values=x_values,
                top_k=None,
                valid_variables=valid_variables,
                step=step
            )

            if not proposals:
                if self._is_main_process():
                    print(f"没有找到有效操作,提前终止")
                break

            # 方案B改进：按位置选择最佳操作，再按操作类型优先级选择
            # 优先级: delete > substitute > keep > insert
            # 这样避免跨操作类型的分数比较（如delete vs keep）
            position_best = {}  # {position: best_proposal}
            for proposal in proposals:
                pos = proposal.position
                if pos not in position_best or proposal.score > position_best[pos].score:
                    position_best[pos] = proposal

            # 收集所有位置的最佳操作，同时应用
            # 这样可以正确处理位置偏移：删除/插入会导致后续位置移动
            all_position_actions = list(position_best.values())

            # 记录每个位置的最佳操作（用于调试和返回）
            if self.logger and self._is_main_process():
                # 传递-1表示没有"选中"的位置（因为所有操作都会执行）
                self.logger.log_position_best_actions(position_best, -1, level=3)

            # 保存本步每个位置的操作详情到历史记录
            step_position_actions = {}
            for pos, prop in position_best.items():
                step_position_actions[pos] = (prop.action_type, prop.score)
            current_candidate.position_actions_history.append(step_position_actions)

            # 构建操作描述（显示所有执行的操作）
            action_descs = []
            for prop in all_position_actions:
                desc = f"{prop.action_type}@{prop.position}"
                if prop.token:
                    desc += f":{prop.token}"
                desc += f"({prop.score:.4f})"
                action_descs.append(desc)

            # Debug: 按action_type排序，让delete操作显示在前面
            action_priority_order = {'delete': 0, 'substitute': 1, 'insert': 2, 'keep': 3}
            action_descs_sorted = sorted(action_descs,
                key=lambda d: action_priority_order.get(d.split('@')[0], 99))

            action_summary = ", ".join(action_descs_sorted[:5])  # 最多显示前5个
            if len(action_descs) > 5:
                action_summary += f" ... (共{len(action_descs)}个操作)"

            history.append(action_summary)

            if self.logger and self._is_main_process():
                print(f"执行操作: {action_summary}")
                # Debug: 显示delete操作
                delete_actions = [prop for prop in all_position_actions if prop.action_type == 'delete']
                if delete_actions:
                    print(f"  Delete操作: {', '.join(f'@{p.position}({p.score:.4f})' for p in delete_actions)}")
                self.logger.log("GREEDY_SEARCH_ACTION",
                               f"step={step} | actions={action_summary} | "
                               f"old_expr={','.join(current_candidate.tokens)}",
                               "greedy_search", level=2)

            # 更新当前候选
            # 使用apply_multiple_actions同时应用所有位置的操作
            # 该方法会通过position_offset正确处理删除/插入导致的位置偏移
            if self.logger and self._is_main_process():
                print(f"  应用操作前: {len(current_candidate.tokens)} tokens")

            new_tokens = self.apply_multiple_actions(
                current_tokens=current_candidate.tokens,
                actions=all_position_actions
            )

            if self.logger and self._is_main_process():
                print(f"  应用操作后: {len(new_tokens)} tokens")
                print(f"  预期token数: 15 (目标表达式)")

            # 计算总分（所有操作的平均分）
            total_score = sum(prop.score for prop in all_position_actions) / len(all_position_actions)

            current_candidate = Candidate(
                tokens=new_tokens,
                score=current_candidate.score + total_score,
                condition=current_candidate.condition,  # 条件保持恒定
                residuals=current_candidate.residuals,   # 残差仅用于日志
                history=history.copy(),
                position_actions_history=current_candidate.position_actions_history.copy()  # 保留位置操作历史
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
