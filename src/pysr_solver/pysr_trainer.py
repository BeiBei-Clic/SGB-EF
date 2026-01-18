import numpy as np
from pysr import PySRRegressor
from sklearn.metrics import r2_score, mean_squared_error


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


def evaluate_model(model: PySRRegressor, X: np.ndarray, y: np.ndarray) -> dict:
    """
    评估 PySR 模型

    Args:
        model: 训练好的 PySRRegressor
        X: 测试集特征
        y: 测试集目标

    Returns:
        评估指标字典
    """
    y_pred = model.predict(X)

    # 获取所有表达式（hall of fame）
    equations = model.equations_
    all_equations = [
        {
            "complexity": int(row["complexity"]),
            "loss": float(row["loss"]),
            "equation": str(row["equation"]),
        }
        for _, row in equations.iterrows()
    ]

    metrics = {
        "r2": r2_score(y, y_pred),
        "mse": mean_squared_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "best_equation": model.get_best()["equation"],
        "best_loss": float(model.get_best()["loss"]),
        "all_equations": all_equations,
    }

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
