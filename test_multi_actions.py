#!/usr/bin/env python
# coding: utf-8

"""
测试多操作同时应用的逻辑

验证场景：
1. 在不同位置插入多个操作
2. 插入和替换混合
3. 插入和删除混合
4. 删除和替换混合
5. 所有操作类型混合
"""

import sys
sys.path.insert(0, '/home/xyh/SGB-EF')

from src.training.greedy_search import ActionProposal


def test_apply_multiple_actions():
    """测试多操作同时应用"""

    print("=" * 60)
    print("测试多操作同时应用逻辑")
    print("=" * 60)

    # 测试场景1：多个插入操作
    print("\n【场景1】多个插入操作")
    print("-" * 60)
    current_tokens = ['x0', 'add', 'x1']
    print(f"原始表达式: {','.join(current_tokens)}")
    print(f"说明: 位置0=x0, 位置1=add, 位置2=x1")

    actions = [
        ActionProposal(action_type='insert', position=1, token='mul', score=0.1, new_tokens=None),
        ActionProposal(action_type='insert', position=2, token='sin', score=0.08, new_tokens=None),
    ]
    print(f"操作: 在位置1插入mul, 在位置2插入sin")

    # 手动应用逻辑（从前往后）
    sorted_actions = sorted(actions, key=lambda a: a.position)
    result_tokens = current_tokens.copy()
    position_offset = 0

    print(f"应用顺序（从前往后）:")
    for i, action in enumerate(sorted_actions):
        print(f"  {i+1}. {action.action_type} 位置{action.position} token={action.token}")

    for i, action in enumerate(sorted_actions):
        print(f"\n步骤{i+1}: 应用 {action.action_type}@{action.position}")
        print(f"  当前表达式: {','.join(result_tokens)}")
        print(f"  当前position_offset: {position_offset}")

        actual_position = action.position + position_offset
        actual_position = max(0, min(actual_position, len(result_tokens)))
        print(f"  计算actual_position: {action.position} + {position_offset} = {actual_position}")

        if action.action_type == 'insert':
            if action.token is not None:
                print(f"  执行insert({actual_position}, '{action.token}')")
                result_tokens.insert(actual_position, action.token)
                position_offset += 1
                print(f"  结果 -> {','.join(result_tokens)}")
                print(f"  更新position_offset: {position_offset}")

    print(f"最终结果: {','.join(result_tokens)}")
    # 从前往后应用：
    # 1. 先应用位置1（在x0之后插入mul）：['x0', 'mul', 'add', 'x1']
    # 2. 再应用位置2（现在add在位置2，在add之后插入sin）：['x0', 'mul', 'add', 'sin', 'x1']
    expected = ['x0', 'mul', 'add', 'sin', 'x1']
    print(f"预期结果: {','.join(expected)}")
    print(f"✓ 通过" if result_tokens == expected else f"✗ 失败")

    # 测试场景2：插入 + 替换（位置会变化）
    print("\n【场景2】插入 + 替换（位置依赖）")
    print("-" * 60)
    current_tokens = ['x0', 'add', 'x1']
    print(f"原始表达式: {','.join(current_tokens)}")

    actions = [
        ActionProposal(action_type='insert', position=1, token='mul', score=0.1, new_tokens=None),
        ActionProposal(action_type='substitute', position=2, token='x2', score=0.09, new_tokens=None),  # 位置2在插入后会变成位置3
    ]

    sorted_actions = sorted(actions, key=lambda a: a.position, reverse=True)
    result_tokens = current_tokens.copy()
    position_offset = 0

    for action in sorted_actions:
        actual_position = action.position + position_offset
        actual_position = max(0, min(actual_position, len(result_tokens)))

        if action.action_type == 'insert':
            if action.token is not None:
                result_tokens.insert(actual_position, action.token)
                position_offset += 1
                print(f"  在位置{actual_position}插入 {action.token} -> {','.join(result_tokens)}")
        elif action.action_type == 'substitute':
            if 0 <= actual_position < len(result_tokens) and action.token is not None:
                old_token = result_tokens[actual_position]
                result_tokens[actual_position] = action.token
                print(f"  位置{actual_position}替换 {old_token}->{action.token} -> {','.join(result_tokens)}")

    print(f"最终结果: {','.join(result_tokens)}")
    # 从后往前应用：先替换位置2的x1->x2，然后在位置1插入mul
    # 原始: ['x0', 'add', 'x1']
    # 替换位置2: ['x0', 'add', 'x2']
    # 插入位置1: ['x0', 'mul', 'add', 'x2']
    expected = ['x0', 'mul', 'add', 'x2']
    print(f"预期结果: {','.join(expected)}")
    print(f"✓ 通过" if result_tokens == expected else "✗ 失败")

    # 测试场景3：插入 + 删除
    print("\n【场景3】插入 + 删除")
    print("-" * 60)
    current_tokens = ['x0', 'add', 'x1', 'mul', 'x2']
    print(f"原始表达式: {','.join(current_tokens)}")
    print(f"说明: 位置0=x0, 位置1=add, 位置2=x1, 位置3=mul, 位置4=x2")

    actions = [
        ActionProposal(action_type='insert', position=2, token='sin', score=0.1, new_tokens=None),
        ActionProposal(action_type='delete', position=3, token=None, score=0.08, new_tokens=None),
    ]
    print(f"操作: 在位置2插入sin, 删除位置3的mul")

    sorted_actions = sorted(actions, key=lambda a: a.position)  # 从前往后
    result_tokens = current_tokens.copy()
    position_offset = 0

    for action in sorted_actions:
        actual_position = action.position + position_offset
        actual_position = max(0, min(actual_position, len(result_tokens)))

        if action.action_type == 'insert':
            if action.token is not None:
                result_tokens.insert(actual_position, action.token)
                position_offset += 1
                print(f"  在位置{actual_position}插入 {action.token} -> {','.join(result_tokens)}")
        elif action.action_type == 'delete':
            if 0 <= actual_position < len(result_tokens) and len(result_tokens) > 1:
                deleted = result_tokens.pop(actual_position)
                position_offset -= 1
                print(f"  删除位置{actual_position}的{deleted} -> {','.join(result_tokens)}")

    print(f"最终结果: {','.join(result_tokens)}")
    # 从前往后应用：
    # 1. 先插入位置2: ['x0', 'add', 'sin', 'x1', 'mul', 'x2']
    # 2. 再删除位置3（现在x1在位置3，删除它）: ['x0', 'add', 'sin', 'mul', 'x2']
    # 等等，这个不对...
    # 让我重新理解：删除位置3的mul
    # 1. 先插入位置2（x1之前插入sin）: ['x0', 'add', 'sin', 'x1', 'mul', 'x2']
    # 2. 再删除位置3+1=4（mul在位置4）: ['x0', 'add', 'sin', 'x1', 'x2']
    expected = ['x0', 'add', 'sin', 'x1', 'x2']
    print(f"预期结果: {','.join(expected)}")
    print(f"✓ 通过" if result_tokens == expected else f"✗ 失败")

    # 测试场景4：复杂场景 - 多个插入、替换、删除混合
    print("\n【场景4】复杂场景 - 多个操作混合")
    print("-" * 60)
    current_tokens = ['x0', 'add', 'x1', 'mul', 'x2']
    print(f"原始表达式: {','.join(current_tokens)}")
    print(f"说明: 位置0=x0, 位置1=add, 位置2=x1, 位置3=mul, 位置4=x2")

    actions = [
        ActionProposal(action_type='insert', position=1, token='sin', score=0.1, new_tokens=None),
        ActionProposal(action_type='substitute', position=3, token='x3', score=0.09, new_tokens=None),
        ActionProposal(action_type='insert', position=4, token='cos', score=0.08, new_tokens=None),
        ActionProposal(action_type='delete', position=2, token=None, score=0.07, new_tokens=None),
    ]
    print(f"操作: 在位置1插入sin, 替换位置3的mul为x3, 在位置4插入cos, 删除位置2的x1")

    sorted_actions = sorted(actions, key=lambda a: a.position)  # 从前往后
    print(f"操作顺序（从前往后）: {[(a.action_type, a.position) for a in sorted_actions]}")

    result_tokens = current_tokens.copy()
    position_offset = 0

    for action in sorted_actions:
        actual_position = action.position + position_offset
        actual_position = max(0, min(actual_position, len(result_tokens)))

        if action.action_type == 'insert':
            if action.token is not None:
                result_tokens.insert(actual_position, action.token)
                position_offset += 1
                print(f"  在位置{actual_position}插入 {action.token} -> {','.join(result_tokens)}")
        elif action.action_type == 'delete':
            if 0 <= actual_position < len(result_tokens) and len(result_tokens) > 1:
                deleted = result_tokens.pop(actual_position)
                position_offset -= 1
                print(f"  删除位置{actual_position}的{deleted} -> {','.join(result_tokens)}")
        elif action.action_type == 'substitute':
            if 0 <= actual_position < len(result_tokens) and action.token is not None:
                old_token = result_tokens[actual_position]
                result_tokens[actual_position] = action.token
                print(f"  位置{actual_position}替换 {old_token}->{action.token} -> {','.join(result_tokens)}")

    print(f"最终结果: {','.join(result_tokens)}")

    # 详细推演：
    # 原始: ['x0', 'add', 'x1', 'mul', 'x2'] (位置0-4)
    # 1. 插入位置1: ['x0', 'sin', 'add', 'x1', 'mul', 'x2'] (offset=1)
    # 2. 删除位置2+1=3: ['x0', 'sin', 'add', 'mul', 'x2'] (offset=0，删除x1)
    # 3. 替换位置3: ['x0', 'sin', 'add', 'x3', 'x2']
    # 4. 插入位置4: ['x0', 'sin', 'add', 'x3', 'cos', 'x2']
    expected = ['x0', 'sin', 'add', 'x3', 'cos', 'x2']
    print(f"预期结果: {','.join(expected)}")
    print(f"✓ 通过" if result_tokens == expected else f"✗ 失败")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    test_apply_multiple_actions()
