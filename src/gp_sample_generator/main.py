"""
主程序：从GP结果生成样本并保存
"""
import json
from pathlib import Path
from src.gp_sample_generator.sample_builder import build_single_sample
from src.pysr_solver.data_loader import load_equation_data
from src.utils.special_tokens import SymbolicRegressionTokenizer


def main():
    # 配置
    GP_RESULT_PATH = "outputs/I.10.7_result.json"
    OUTPUT_PATH = "data/gp_samples_I.10.7.txt"
    N_SAMPLES = 100  # 每个表达式采样100个数据点
    MAX_EXPR_LENGTH = 24

    # 加载GP结果
    with open(GP_RESULT_PATH) as f:
        gp_data = json.load(f)
    equation_name = gp_data['equation']

    # 加载Feynman数据
    X, y_true, metadata = load_equation_data(equation_name, n_samples=N_SAMPLES)

    # 初始化tokenizer
    max_dim = X.shape[1]
    tokenizer = SymbolicRegressionTokenizer(max_dim=max_dim)

    # 获取表达式对
    target_expr = gp_data['best_equation']
    pairs = [(eq_info['equation'], target_expr) for eq_info in gp_data['all_equations']]

    # 生成样本
    samples = []
    for curr_expr, target_expr in pairs:
        try:
            sample = build_single_sample(
                curr_expr, target_expr, X, y_true, tokenizer, MAX_EXPR_LENGTH
            )
            samples.append(sample)
        except Exception as e:
            print(f"跳过表达式 {curr_expr[:30]}...: {e}")
            continue

    # 保存
    with open(OUTPUT_PATH, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"生成 {len(samples)} 个样本，保存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
