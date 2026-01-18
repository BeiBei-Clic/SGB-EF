import numpy as np
import pandas as pd
from pathlib import Path


# 获取项目根目录
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_RESOURCES_DIR = _PROJECT_ROOT / "resources"


def load_equation_data(equation_name: str, n_samples: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    加载指定方程的数据

    Args:
        equation_name: 方程名称 (如 I.10.7)
        n_samples: 采样数量，None 表示使用全部数据

    Returns:
        X: 输入特征数组 (n_samples, n_features)
        y: 输出目标数组 (n_samples,)
        metadata: 方程元数据字典
    """
    # 加载元数据
    df_meta = pd.read_csv(_RESOURCES_DIR / "FeynmanEquations.csv")
    eq_row = df_meta[df_meta["Filename"] == equation_name]

    if eq_row.empty:
        raise ValueError(f"方程 {equation_name} 不存在于元数据中")

    eq_row = eq_row.iloc[0]
    metadata = {
        "filename": eq_row["Filename"],
        "formula": eq_row["Formula"],
        "n_variables": eq_row["# variables"],
    }

    # 提取变量信息
    var_names = []
    for i in range(1, 11):
        v_name = eq_row.get(f"v{i}_name")
        if pd.notna(v_name):
            var_names.append(v_name)
        else:
            break

    metadata["var_names"] = var_names

    # 加载数据文件
    data_path = _RESOURCES_DIR / f"Feynman_with_units/{equation_name}"
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    # 读取数据（空格分隔）
    data = np.loadtxt(data_path)

    # 最后一列是目标值，其余是特征
    X = data[:, :-1]
    y = data[:, -1]

    # 采样
    if n_samples is not None and n_samples < len(X):
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y, metadata


if __name__ == "__main__":
    # 测试数据加载
    X, y, meta = load_equation_data("I.10.7", n_samples=1000)
    print(f"公式: {meta['formula']}")
    print(f"变量: {meta['var_names']}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X 前3行:\n{X[:3]}")
    print(f"y 前3个: {y[:3]}")
