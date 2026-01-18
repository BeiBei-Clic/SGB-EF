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
        position: 位置索引（位置0=BOS，位置1,2,3...=序列中的后续token）
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
                 numerical_clip_threshold=1e6,
                 num_inference_timesteps: int = 10):
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
            num_inference_timesteps: 推理时使用的时间步数量
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

        # 准备推理时间步序列（始终启用）
        from ..core.sampling import TimestepSampler
        self.timestep_sampler = TimestepSampler(
            sampling_strategy="discrete",
            num_discrete_timesteps=num_inference_timesteps
        )
        self.inference_timesteps = self.timestep_sampler.get_timesteps_for_inference(
            num_inference_timesteps, device
        )
        self.current_timestep_idx = 0
        self.num_inference_timesteps = num_inference_timesteps

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

        # 准备时间步（使用当前推理步对应的时间步）
        timestep = self.inference_timesteps[self.current_timestep_idx:self.current_timestep_idx+1]

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                condition=condition,
                timestep=timestep
            )

            # 提取结果（使用logits，和训练时一致）
            rates_logits = output['rates_logits']  # [1, seq_len, 4]
            insert_logits = output['insert_logits']  # [1, seq_len, vocab_size]
            substitute_logits = output['substitute_logits']  # [1, seq_len, vocab_size]

            # 计算操作概率（仅用于日志记录）
            rates_probs = F.softmax(rates_logits, dim=-1)
            lambda_ins = rates_probs[0, :, 0].cpu().numpy()  # 插入概率
            lambda_del = rates_probs[0, :, 1].cpu().numpy()  # 删除概率
            lambda_sub = rates_probs[0, :, 2].cpu().numpy()  # 替换概率
            lambda_keep = rates_probs[0, :, 3].cpu().numpy()  # 保持概率

            # 计算token概率（仅用于日志记录）
            insert_probs = F.softmax(insert_logits, dim=-1)
            substitute_probs = F.softmax(substitute_logits, dim=-1)

            effective_length = int(attention_mask[0].sum().item())

            # 获取词汇表大小
            vocab_size = self.tokenizer.vocab_size

            # 为每个位置选择最佳操作（在联合logit空间，和训练时完全一致）
            for pos in range(len(current_tokens) + 1):
                input_ids_pos = pos  # 位置索引（位置0=BOS，位置1,2,3...=后续token）

                if input_ids_pos >= rates_logits.shape[1]:
                    continue

                # 获取当前位置的token信息（位置0是BOS token）
                token_id = None
                token_name = None
                if input_ids_pos == 0:
                    token_name = '<BOS>'
                    token_id = self.tokenizer.convert_tokens_to_ids('<s>')
                elif 0 <= (input_ids_pos - 1) < len(current_tokens):
                    token_name = current_tokens[input_ids_pos - 1]
                    token_id = self.tokenizer.convert_tokens_to_ids(token_name)

                # 记录token位置映射和模型预测
                if self.logger and self._is_main_process() and token_name is not None:
                    self.logger.log_token_position_mapping(
                        step=step,
                        position=input_ids_pos,
                        token_id=token_id,
                        token_name=token_name,
                        lambda_ins=lambda_ins[pos] if pos < len(lambda_ins) else 0.0,
                        lambda_del=lambda_del[pos] if pos < len(lambda_del) else 0.0,
                        lambda_sub=lambda_sub[pos] if pos < len(lambda_sub) else 0.0,
                        lambda_keep=lambda_keep[pos] if pos < len(lambda_keep) else 0.0
                    )

                # 重构训练时的logit空间（editflow_manager.py:460-467）
                # ins_logits = insert_logits + ins_logits_rate
                ins_logits = insert_logits[0, input_ids_pos] + rates_logits[0, input_ids_pos, 0]
                # del_logits = del_logits_rate (DELETE无token logits)
                del_logits = rates_logits[0, input_ids_pos, 1]
                # sub_logits = substitute_logits + sub_logits_rate
                sub_logits = substitute_logits[0, input_ids_pos] + rates_logits[0, input_ids_pos, 2]
                # keep_logits = keep_logits_rate (KEEP无token logits)
                keep_logits = rates_logits[0, input_ids_pos, 3]

                # 拼接所有操作logits：[INSERT...v] [DELETE] [SUBSTITUTE...v] [KEEP]
                all_logits = torch.cat([ins_logits, del_logits.unsqueeze(0),
                                        sub_logits, keep_logits.unsqueeze(0)])
                # 维度: [2*vocab_size+2]

                # Softmax得到概率分布
                all_probs = F.softmax(all_logits, dim=0)

                # 找到最大概率的操作
                best_op_idx = all_logits.argmax()
                best_prob = all_probs[best_op_idx].item()

                # 根据lambda值确定最佳操作（用于日志）
                if lambda_del[pos] > lambda_ins[pos] and lambda_del[pos] > lambda_sub[pos] and lambda_del[pos] > lambda_keep[pos]:
                    best_action = "DELETE"
                elif lambda_ins[pos] > lambda_keep[pos] and lambda_ins[pos] > lambda_sub[pos]:
                    best_action = "INSERT"
                elif lambda_sub[pos] > lambda_keep[pos]:
                    best_action = "SUBSTITUTE"
                else:
                    best_action = "KEEP"

                # 记录位置预测
                if self.logger and self._is_main_process() and token_name is not None:
                    self.logger.log_position_prediction(
                        step=step,
                        pos=pos,
                        token_name=token_name if token_name != '<BOS>' else 'BOS',
                        lambda_ins=lambda_ins[pos] if pos < len(lambda_ins) else 0.0,
                        lambda_del=lambda_del[pos] if pos < len(lambda_del) else 0.0,
                        lambda_sub=lambda_sub[pos] if pos < len(lambda_sub) else 0.0,
                        lambda_keep=lambda_keep[pos] if pos < len(lambda_keep) else 0.0,
                        best_action=best_action
                    )

                # 解码操作类型和token
                if best_op_idx < vocab_size:
                    # INSERT操作
                    token_id = int(best_op_idx)  # 转换为int
                    # 直接使用ids_to_tokens，避免PreTrainedTokenizer的convert_ids_to_tokens干扰
                    token_name = self.tokenizer.ids_to_tokens.get(token_id, '<unk>')

                    # 变量验证
                    if valid_variables is not None and token_name.startswith('x') and token_name not in valid_variables:
                        continue

                    # 生成新tokens
                    if pos == 0:
                        new_tokens = current_tokens.copy()
                        new_tokens.insert(0, token_name)
                    else:
                        new_tokens = current_tokens.copy()
                        new_tokens.insert(pos, token_name)

                    proposals.append(ActionProposal(
                        action_type='insert',
                        position=input_ids_pos,
                        token=token_name,
                        score=best_prob,  # 使用联合概率
                        new_tokens=new_tokens
                    ))

                    # 记录INSERT操作
                    if self.logger and self._is_main_process():
                        self.logger.log_action_execution(
                            step=step,
                            position=input_ids_pos,
                            action_type='insert',
                            token_name=token_name,
                            score=best_prob
                        )

                elif best_op_idx == vocab_size:
                    # DELETE操作
                    if len(current_tokens) > 1:
                        current_token_idx = input_ids_pos - 1
                        if 0 <= current_token_idx < len(current_tokens):
                            deleted_token = current_tokens[current_token_idx]
                            new_tokens = current_tokens.copy()
                            del new_tokens[current_token_idx]

                            proposals.append(ActionProposal(
                                action_type='delete',
                                position=input_ids_pos,
                                score=best_prob,  # 使用联合概率
                                new_tokens=new_tokens
                            ))

                            # 记录DELETE操作
                            if self.logger and self._is_main_process():
                                self.logger.log_action_execution(
                                    step=step,
                                    position=input_ids_pos,
                                    action_type='delete',
                                    token_name=deleted_token,
                                    score=best_prob
                                )

                            # 记录删除后的位置变化
                            if self.logger and self._is_main_process():
                                self.logger.log_position_after_deletion(
                                    step=step,
                                    deleted_pos=input_ids_pos,
                                    old_tokens=current_tokens,
                                    new_tokens=new_tokens
                                )

                elif best_op_idx < 2 * vocab_size + 1:
                    # SUBSTITUTE操作
                    token_id = int(best_op_idx - vocab_size - 1)  # 转换为int

                    # 直接使用ids_to_tokens，避免PreTrainedTokenizer的convert_ids_to_tokens干扰
                    token_name = self.tokenizer.ids_to_tokens.get(token_id, '<unk>')

                    # 变量验证
                    if valid_variables is not None and token_name.startswith('x') and token_name not in valid_variables:
                        continue

                    current_token_idx = input_ids_pos - 1
                    if 0 <= current_token_idx < len(current_tokens):
                        old_token = current_tokens[current_token_idx]
                        new_tokens = current_tokens.copy()
                        new_tokens[current_token_idx] = token_name

                        proposals.append(ActionProposal(
                            action_type='substitute',
                            position=input_ids_pos,
                            token=token_name,
                            score=best_prob,  # 使用联合概率
                            new_tokens=new_tokens
                        ))

                        # 记录SUBSTITUTE操作
                        if self.logger and self._is_main_process():
                            self.logger.log_action_execution(
                                step=step,
                                position=input_ids_pos,
                                action_type='substitute',
                                token_name=f"{old_token}→{token_name}",
                                score=best_prob
                            )

                else:
                    # KEEP操作
                    current_token_idx = input_ids_pos - 1
                    if 0 <= current_token_idx < len(current_tokens):
                        kept_token = current_tokens[current_token_idx]
                        proposals.append(ActionProposal(
                            action_type='keep',
                            position=input_ids_pos,
                            score=best_prob,  # 使用联合概率
                            new_tokens=current_tokens.copy()
                        ))

                        # 记录KEEP操作
                        if self.logger and self._is_main_process():
                            self.logger.log_action_execution(
                                step=step,
                                position=input_ids_pos,
                                action_type='keep',
                                token_name=kept_token,
                                score=best_prob
                            )

        # 按分数排序
        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals

    def apply_multiple_actions(self,
                               current_tokens: List[str],
                               actions: List[ActionProposal]) -> List[str]:
        """同时应用多个操作到表达式，智能处理操作间的位置依赖关系

        位置说明：位置0=BOS token，位置1,2,3...=序列中的实际token
        current_tokens不包含BOS，但action.position使用的是包含BOS的位置索引

        核心思路：
        1. 操作按原始位置从前往后应用
        2. 每次操作后，动态更新后续操作的位置
        3. 插入操作使后续位置+1，删除操作使后续位置-1

        Args:
            current_tokens: 当前token列表（不含BOS）
            actions: 要应用的操作列表（应该已经按阈值过滤）

        Returns:
            应用所有操作后的新token列表（不含BOS）
        """
        if not actions:
            return current_tokens.copy()

        if len(actions) == 1:
            # 单个操作，直接返回
            return actions[0].new_tokens.copy()

        # 多个操作：需要智能处理位置依赖
        # 策略：DELETE操作从后往前执行（避免位置偏移），其他操作从前往后执行

        # 1. 按位置升序排序
        sorted_actions = sorted(actions, key=lambda a: a.position)

        # 2. 初始化：从当前表达式开始
        result_tokens = current_tokens.copy()

        # 3. 将操作分为 DELETE 和非 DELETE 两类
        delete_actions = [a for a in sorted_actions if a.action_type == 'delete']
        other_actions = [a for a in sorted_actions if a.action_type != 'delete']

        # 4. 执行DELETE操作（从后往前）
        result_tokens = self._execute_delete_actions(result_tokens, delete_actions)

        # 5. 执行其他操作（从前往后）
        position_offset = -len(delete_actions)
        result_tokens = self._execute_other_actions(
            result_tokens, other_actions, position_offset
        )

        return result_tokens

    def _execute_delete_actions(self, result_tokens: List[str],
                               delete_actions: List[ActionProposal]) -> List[str]:
        """执行DELETE操作（从后往前，避免位置偏移）

        Args:
            result_tokens: 当前token列表
            delete_actions: DELETE操作列表

        Returns:
            执行DELETE后的token列表
        """
        # Debug: 打印初始状态
        if os.environ.get('RANK', '0') == '0':
            print(f"    [DEBUG REVERSE DELETE START] result_tokens={result_tokens}, delete_actions={[a.position for a in delete_actions]}")

        for i, action in enumerate(reversed(delete_actions)):
            # DELETE操作：position指向input_ids位置（位置0=BOS，位置1,2,3...=实际token）
            # 需要转换为current_tokens位置：pos-1（因为current_tokens不包含BOS）
            actual_position = (action.position - 1)

            # 确保位置在有效范围内
            if actual_position < 0:
                actual_position = 0
            if actual_position >= len(result_tokens):
                actual_position = len(result_tokens) - 1

            # 执行删除
            if 0 <= actual_position < len(result_tokens) and len(result_tokens) > 1:
                deleted_token = result_tokens[actual_position]
                del result_tokens[actual_position]
                # Debug: 只在main process打印
                if os.environ.get('RANK', '0') == '0':
                    print(f"    [DEBUG REVERSE DELETE {i+1}/{len(delete_actions)}] @{action.position}→actual@{actual_position}: '{deleted_token}', 剩余{result_tokens}")

        if os.environ.get('RANK', '0') == '0':
            print(f"    [DEBUG REVERSE DELETE END] result_tokens={result_tokens}")

        return result_tokens

    def _execute_other_actions(self, result_tokens: List[str],
                              other_actions: List[ActionProposal],
                              position_offset: int) -> List[str]:
        """执行非DELETE操作（从前往后）

        Args:
            result_tokens: 当前token列表
            other_actions: 非DELETE操作列表
            position_offset: 初始位置偏移量

        Returns:
            执行操作后的token列表
        """
        for action in other_actions:
            # 计算实际位置：将input_ids位置转换为current_tokens位置
            if action.action_type == 'insert':
                # INSERT操作：position表示在哪个input_ids位置之后插入
                if action.position == 0:
                    actual_position = 0 + position_offset
                else:
                    actual_position = action.position + position_offset
            else:
                # SUBSTITUTE/KEEP操作：position指向input_ids位置
                actual_position = (action.position - 1) + position_offset

            # 确保位置在有效范围内
            actual_position = max(0, min(actual_position, len(result_tokens)))

            if action.action_type == 'insert':
                # 插入操作
                if action.token is not None:
                    if (0 <= actual_position < len(result_tokens) and
                        result_tokens[actual_position] == '<gap>'):
                        # 如果是gap，直接替换
                        result_tokens[actual_position] = action.token
                    else:
                        # 如果不是gap，插入
                        result_tokens.insert(actual_position, action.token)
                        position_offset += 1  # 后续位置后移

            elif action.action_type == 'substitute':
                # 替换操作
                if 0 <= actual_position < len(result_tokens) and action.token is not None:
                    result_tokens[actual_position] = action.token

            elif action.action_type == 'keep':
                # KEEP操作：保持不变
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

            from ...symbolic.symbolic_utils import (
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
            print(f"使用多时间步推理模式: {self.num_inference_timesteps} 个时间步")
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

            # 多时间步推理：基于表达式长度的自适应时间步调度
            # 方案：根据当前表达式长度与目标长度的比例来估计时间步
            current_length = len(current_candidate.tokens)
            # 假设目标长度约15（可根据实际情况调整）
            target_length_estimate = 15
            # t=1表示简单（短），t=0表示复杂（长）
            estimated_t = 1.0 - min(current_length / target_length_estimate, 1.0)

            # 平滑过渡：结合推理进度
            t_progress = step / n_steps
            current_t = 0.7 * estimated_t + 0.3 * t_progress  # 加权融合

            self.current_timestep_idx = min(
                int((1.0 - current_t) * self.num_inference_timesteps),
                self.num_inference_timesteps - 1
            )

            if self._is_main_process():
                print(f"  当前时间步: {current_t:.3f} (基于表达式长度: {current_length})")

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

            # 按位置选择最佳操作（联合logit空间已经让所有操作公平竞争）
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
                               "inference", level=3)
        else:
            if self.logger and self._is_main_process():
                print(f"无法评估最终表达式的MSE")
                self.logger.log("GREEDY_SEARCH_FAILED",
                               f"无法评估最终表达式 | tokens={','.join(current_candidate.tokens)}",
                               "inference", level=3)

        return current_candidate

    def _is_main_process(self):
        """检查是否为主进程"""
        return True
