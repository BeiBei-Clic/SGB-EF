import sys
from pathlib import Path
import numpy as np
from pysr import PySRRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sympy import sympify

# 添加项目根目录到 path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

from src.symbolic.symbolic_utils import simplify_expr


def train_pysr(X: np.ndarray, y: np.ndarray, niterations: int = 100, **kwargs) -> PySRRegressor:
    """
    使用 PySR 训练符号回归模型

    Args:
        X: 输入特征数组
        y: 输出目标数组
        niterations: 迭代次数
        **kwargs: 其他 PySR 参数

    Returns:
        训练好的 PySRRegressor 模型
    """
    default_params = {
        "binary_operators": ["+", "-", "*", "/", "^"],
        "unary_operators": [
            "sqrt", "exp", "log", "abs",
            "sin", "cos", "tan",
            "asin", "acos", "atan",
            "sinh", "cosh", "tanh",
        ],
        "maxsize": 30,
        "niterations": niterations,
        "populations": 15,
        "ncycles_per_iteration": 500,
    }

    # 合并用户参数
    params = {**default_params, **kwargs}

    model = PySRRegressor(**params)
    model.fit(X, y)

    return model


def evaluate_model(
    model: PySRRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> dict:
    """
    评估 PySR 模型

    Args:
        model: 训练好的 PySRRegressor
        X_train: 训练集特征
        y_train: 训练集目标
        X_test: 测试集特征（可选，如果不提供则用训练集评估）
        y_test: 测试集目标（可选）

    Returns:
        评估指标字典
    """
    # 在测试集上预测（如果没有测试集则用训练集）
    X_eval = X_test if X_test is not None else X_train
    y_eval = y_test if y_test is not None else y_train

    y_pred = model.predict(X_eval)

    # 获取所有表达式（hall of fame）
    equations = model.equations_
    all_equations = [
        {
            "complexity": int(row["complexity"]),
            "loss": float(row["loss"]),
            "equation": str(simplify_expr(sympify(str(row["equation"]).replace('^', '**')))),
        }
        for _, row in equations.iterrows()
    ]

    best = model.get_best()

    # 计算评估集指标
    eval_metrics = {
        "r2": r2_score(y_eval, y_pred),
        "mse": mean_squared_error(y_eval, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_eval, y_pred)),
    }

    # 如果有测试集，也计算训练集指标（用于检测过拟合）
    if X_test is not None:
        y_pred_train = model.predict(X_train)
        train_metrics = {
            "r2_train": r2_score(y_train, y_pred_train),
            "mse_train": mean_squared_error(y_train, y_pred_train),
            "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        }
        metrics = {**train_metrics, **eval_metrics}
    else:
        metrics = eval_metrics

    metrics.update({
        "best_equation": str(simplify_expr(sympify(best["equation"].replace('^', '**')))),
        "best_loss": float(best["loss"]),
        "all_equations": all_equations,
    })

    return metrics


if __name__ == "__main__":
    # 测试训练
    from data_loader import load_equation_data

    X, y, meta = load_equation_data("I.10.7", n_samples=1000)

    print(f"训练方程: {meta['formula']}")
    print(f"变量: {meta['var_names']}")
    print(f"数据量: {len(X)}")

    model = train_pysr(X, y, niterations=5)

    metrics = evaluate_model(model, X, y)
    print(f"\n最佳表达式: {metrics['best_equation']}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"MSE: {metrics['mse']:.4e}")
