#!/usr/bin/env python3
"""
调试数据生成器的脚本
"""

import time
import traceback
import signal
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.symbolic.data_generator import generate_triplet_samples

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def test_data_generation():
    print("开始测试数据生成...")

    # 设置超时信号（只适用于Unix系统）
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60秒超时

    try:
        start_time = time.time()

        # 先测试小样本
        print("测试10个样本...")
        samples = generate_triplet_samples(
            num_samples=10,
            max_dim=3,
            n_points=50,
            max_depth=3
        )

        elapsed = time.time() - start_time
        print(f"10个样本生成完成，耗时: {elapsed:.2f}秒")
        print(f"平均每个样本: {elapsed/10:.2f}秒")

        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # 取消超时

        return True

    except TimeoutError:
        print("❌ 数据生成超时（60秒）！存在死循环或性能问题")
        return False
    except Exception as e:
        print(f"❌ 数据生成出错: {e}")
        traceback.print_exc()
        return False

def test_individual_components():
    """测试各个组件"""
    print("\n测试各个组件...")

    try:
        from src.symbolic.data_generator import (
            generate_random_expr,
            expr_to_tree,
            corrupt_expression,
            evaluate_expr,
            compute_expression_alignment,
            levenshtein_alignment
        )
        import sympy as sp
        import numpy as np

        # 测试表达式生成
        print("1. 测试表达式生成...")
        expr = generate_random_expr(2, max_depth=3)
        print(f"   生成表达式: {expr}")

        # 测试表达式转树
        print("2. 测试表达式转树...")
        tree_str = expr_to_tree(expr)
        print(f"   树结构: {tree_str}")

        # 测试表达式破坏
        print("3. 测试表达式破坏...")
        corrupted = corrupt_expression(expr, 0.5)
        print(f"   破坏后: {corrupted}")

        # 测试表达式求值
        print("4. 测试表达式求值...")
        x_values = np.random.uniform(-2, 2, (20, 2))
        y_vals = evaluate_expr(expr, x_values)
        print(f"   求值结果形状: {y_vals.shape}")

        # 测试对齐计算
        print("5. 测试对齐计算...")
        alignment = compute_expression_alignment(corrupted, expr)
        print(f"   对齐长度: {len(alignment['alignment'])}")
        print(f"   编辑距离: {alignment['edit_distance']}")

        # 测试Levenshtein对齐（重点测试）
        print("6. 测试Levenshtein对齐...")
        tokens1 = expr_to_tree(expr).split(',')
        tokens2 = expr_to_tree(corrupted).split(',')
        print(f"   tokens1长度: {len(tokens1)}")
        print(f"   tokens2长度: {len(tokens2)}")

        start_time = time.time()
        lev_alignment = levenshtein_alignment(tokens1, tokens2)
        lev_time = time.time() - start_time
        print(f"   Levenshtein对齐耗时: {lev_time:.4f}秒")
        print(f"   对齐操作数: {len(lev_alignment)}")

        return True

    except Exception as e:
        print(f"❌ 组件测试出错: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 数据生成器调试脚本 ===")

    # 测试各个组件
    component_ok = test_individual_components()

    if component_ok:
        print("\n✅ 所有组件测试通过")

        # 测试完整数据生成
        print("\n开始完整数据生成测试...")
        data_ok = test_data_generation()

        if data_ok:
            print("\n✅ 数据生成测试通过")
        else:
            print("\n❌ 数据生成测试失败")
    else:
        print("\n❌ 组件测试失败，跳过完整测试")