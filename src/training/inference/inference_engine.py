"""
推理引擎 - 专注于符号回归推理
"""

import torch
import sympy as sp


class InferenceEngine:
    """EditFlow推理引擎 - 负责符号回归推理"""

    # 类常量：推理配置参数
    MIN_ACTION_SCORE = 0.01
    MAX_EXPRESSION_LENGTH = 50
    NUMERICAL_CLIP_THRESHOLD = 1e6

    def __init__(self, model, condition_encoder, tokenizer, args, logger, device):
        """
        初始化推理引擎

        Args:
            model: EditFlow模型（eval模式）
            condition_encoder: 条件编码器（eval模式）
            tokenizer: 分词器
            args: 配置参数
            logger: 日志记录器
            device: 运行设备
        """
        self.model = model
        self.condition_encoder = condition_encoder
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger
        self.device = device

        # 确保模型处于评估模式
        self.model.eval()
        self.condition_encoder.eval()

        # 导入搜索器（延迟导入避免循环依赖）
        from ..greedy_search import SimpleSymbolicRegression
        self.searcher = SimpleSymbolicRegression(
            model=model,
            condition_encoder=condition_encoder,
            tokenizer=tokenizer,
            device=device,
            args=args,
            logger=logger,
            min_action_score=self.MIN_ACTION_SCORE,
            max_expression_length=self.MAX_EXPRESSION_LENGTH,
            numerical_clip_threshold=self.NUMERICAL_CLIP_THRESHOLD
        )

    def _prepare_initial_expression(self, initial_expr, x_data, y_data_len):
        """准备初始表达式和tokens

        Args:
            initial_expr: 初始表达式（sympy表达式、字符串或token列表）
            x_data: 输入x数据
            y_data_len: y数据长度

        Returns:
            tuple: (initial_expr, current_tokens, y_pred)
        """
        from ...symbolic.symbolic_utils import evaluate_expression_safe, expr_to_tree

        # 处理初始表达式
        if isinstance(initial_expr, list):
            current_tokens = initial_expr
            initial_expr = None
        else:
            if initial_expr is None:
                initial_expr = sp.Symbol('x0')
            elif isinstance(initial_expr, str):
                initial_expr = sp.sympify(initial_expr)

            initial_expr_str = expr_to_tree(initial_expr)
            current_tokens = initial_expr_str.split(',') if initial_expr_str else ['x0']

        # 计算初始表达式的预测值
        if initial_expr is not None:
            success, y_pred = evaluate_expression_safe(initial_expr, x_data)
            if not success:
                self.logger.log("INITIAL_EXPR_WARN", "无法计算初始表达式的预测值，使用零初始化", "inference", level=3)
                y_pred = [0.0] * y_data_len
        else:
            y_pred = [0.0] * y_data_len

        return initial_expr, current_tokens, y_pred

    def symbolic_regression(self, x_data, y_data, condition, x_values, y_values,
                           n_steps=100, initial_expr=None):
        """执行符号回归推理

        Args:
            x_data: 输入x数据 (numpy array)
            y_data: 目标y数据 (numpy array)
            condition: 编码后的条件张量
            x_values: x数据的张量形式 (torch.Tensor)
            y_values: y数据的张量形式 (torch.Tensor)
            n_steps: 最大推理步数
            initial_expr: 初始表达式（sympy表达式或字符串），如果为None则使用x0

        Returns:
            dict: 推理结果，包含final_expression, initial_tokens, final_tokens, history等
        """
        self.logger.log("SYMBOLIC_REGRESSION_START",
                       f"输入数据: x形状={x_data.shape}, y形状={y_data.shape} | n_steps={n_steps}",
                       "inference", level=3)

        # 准备初始表达式
        initial_expr, current_tokens, y_pred = self._prepare_initial_expression(
            initial_expr, x_data, len(y_data)
        )

        # 计算残差
        residuals = y_values - torch.FloatTensor(y_pred).unsqueeze(0).to(self.device)

        # 记录初始信息
        self.logger.log("INITIAL_DATA",
                       f"x_values: shape={x_values.shape} range=[{x_values.min():.4f},{x_values.max():.4f}] | "
                       f"y_target: shape={y_values.shape} range=[{y_values.min():.4f},{y_values.max():.4f}] | "
                       f"residuals: shape={residuals.shape} range=[{residuals.min():.4f},{residuals.max():.4f}] | "
                       f"initial_expr: {initial_expr} | initial_tokens: {current_tokens}",
                       "inference", level=3)
        self.logger.log("ARCHITECTURE_INFO",
                       "使用目标值y_target作为条件（架构改进：北极星模式）",
                       "inference", level=3)

        # 打印条件嵌入的前10个维度
        condition_cpu = condition.cpu().squeeze(0)
        condition_values = condition_cpu.detach().numpy()
        condition_preview = condition_values.flatten()[:10] if condition_values.ndim == 2 else condition_values[:10]
        self.logger.log("INITIAL_CONDITION",
                       f"condition: shape={condition.shape} range=[{condition.min():.4f},{condition.max():.4f}] | "
                       f"前10维: [{', '.join([f'{float(v):.6f}' for v in condition_preview])}]",
                       "inference", level=3)

        # 执行推理
        print(f"\n执行单最佳操作推理...")

        residuals_np = residuals.cpu().squeeze(0).numpy()
        best_candidate = self.searcher.greedy_search(
            initial_tokens=current_tokens,
            initial_condition=condition,
            initial_residuals=residuals_np,
            x_data=x_data,
            y_data=y_data,
            x_values=x_values,
            n_steps=n_steps
        )

        final_expression = ','.join(best_candidate.tokens) if best_candidate and best_candidate.tokens else ""

        if best_candidate:
            mse_score = best_candidate.mse_score
            mse_str = f'{mse_score:.6f}' if mse_score is not None else 'N/A'
            self.logger.log("SIMPLE_SEARCH_RESULT",
                           f"MSE分数: {mse_str} | "
                           f"操作历史: {' -> '.join(best_candidate.history[-5:]) if best_candidate.history else 'N/A'}",
                           "inference", level=3)

        self.logger.log("INFERENCE_COMPLETE", f"最终表达式: {final_expression}", "inference", level=3)

        return {
            'final_expression': final_expression,
            'initial_tokens': current_tokens,
            'final_tokens': best_candidate.tokens if best_candidate else [],
            'history': best_candidate.history if best_candidate else [],
            'position_actions_history': best_candidate.position_actions_history if best_candidate else [],
            'mse_score': best_candidate.mse_score if best_candidate else None
        }
