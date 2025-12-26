"""
测试LlamaEditFlowBackbone模型的基本功能
"""
import torch
import sys
from pathlib import Path

# 添加项目路径到sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.modeling.llama_editflow import LlamaEditFlowBackbone, LlamaEditFlowConfig
from src.utils.special_tokens import SpecialTokensManager


def create_symbolic_vocab(max_dim=10):
    """创建符号回归专用词表"""
    # 运算符: 5个
    operators = ['add', 'sub', 'mul', 'div', 'pow']

    # 函数: 7个
    functions = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']

    # 变量: x0-x9
    variables = [f'x{i}' for i in range(max_dim)]

    # 特殊token
    special_tokens = ['<gap>', 'constant', '<s>', '</s>', '<pad>', '<unk>', '<mask>']

    # 合并所有token
    all_tokens = operators + functions + variables + special_tokens

    # 创建token到索引的映射
    vocab = {token: idx for idx, token in enumerate(all_tokens)}

    return vocab


def test_llama_editflow_basic():
    """测试基本的模型初始化和前向传播"""
    print("=" * 60)
    print("测试 LlamaEditFlowBackbone 基本功能")
    print("=" * 60)

    # 1. 创建词表
    vocab = create_symbolic_vocab(max_dim=10)
    vocab_size = len(vocab)
    print(f"\n1. 词表大小: {vocab_size}")
    print(f"   示例tokens: {list(vocab.keys())[:10]}")

    # 2. 配置模型
    config = LlamaEditFlowConfig(
        vocab_size=vocab_size,
        hidden_dim=128,  # 较小的维度用于测试
        n_layers=2,      # 较少的层用于测试
        n_heads=4,
        condition_dim=128,
        dropout=0.1,
        max_seq_len=24,
        use_condition_injection=True,
    )

    print(f"\n2. 模型配置:")
    print(f"   隐藏层维度: {config.hidden_dim}")
    print(f"   层数: {config.n_layers}")
    print(f"   注意力头数: {config.n_heads}")

    # 3. 初始化模型
    print(f"\n3. 初始化模型...")
    model = LlamaEditFlowBackbone(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        condition_dim=config.condition_dim,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        use_condition_injection=config.use_condition_injection,
        verbose=True
    )

    # 4. 创建测试输入
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    time_steps = torch.rand(batch_size, 1)
    condition = torch.randn(batch_size, 32, config.condition_dim)  # 32个条件token
    attention_mask = torch.ones(batch_size, seq_len)

    print(f"\n4. 创建测试输入:")
    print(f"   input_ids shape: {input_ids.shape}")
    print(f"   time_steps shape: {time_steps.shape}")
    print(f"   condition shape: {condition.shape}")
    print(f"   attention_mask shape: {attention_mask.shape}")

    # 5. 前向传播
    print(f"\n5. 执行前向传播...")
    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                time_steps=time_steps,
                condition=condition,
                attention_mask=attention_mask
            )

        print(f"   ✓ 前向传播成功!")

        # 6. 验证输出
        print(f"\n6. 验证输出:")
        ins_rate, del_rate, sub_rate = output['rates']
        print(f"   插入率 shape: {ins_rate.shape} (期望: [{batch_size}, {seq_len}, 1])")
        print(f"   删除率 shape: {del_rate.shape} (期望: [{batch_size}, {seq_len}, 1])")
        print(f"   替换率 shape: {sub_rate.shape} (期望: [{batch_size}, {seq_len}, 1])")

        print(f"   插入logits shape: {output['insert_logits'].shape} (期望: [{batch_size}, {seq_len}, {vocab_size}])")
        print(f"   替换logits shape: {output['substitute_logits'].shape} (期望: [{batch_size}, {seq_len}, {vocab_size}])")

        print(f"   插入概率 shape: {output['insert_probs'].shape} (期望: [{batch_size}, {seq_len}, {vocab_size}])")
        print(f"   替换概率 shape: {output['substitute_probs'].shape} (期望: [{batch_size}, {seq_len}, {vocab_size}])")

        # 7. 验证值的范围
        print(f"\n7. 验证值的范围:")
        print(f"   插入率范围: [{ins_rate.min():.4f}, {ins_rate.max():.4f}] (应为正数)")
        print(f"   删除率范围: [{del_rate.min():.4f}, {del_rate.max():.4f}] (应为正数)")
        print(f"   替换率范围: [{sub_rate.min():.4f}, {sub_rate.max():.4f}] (应为正数)")

        print(f"   插入概率和: {output['insert_probs'][0, 0].sum().item():.4f} (应接近1.0)")
        print(f"   替换概率和: {output['substitute_probs'][0, 0].sum().item():.4f} (应接近1.0)")

        # 8. 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n8. 模型参数量:")
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数量: {trainable_params:,}")

        print(f"\n{'=' * 60}")
        print(f"✓ 所有测试通过!")
        print(f"{'=' * 60}")

        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """测试向后兼容的EditFlowTransformer包装类"""
    print("\n" + "=" * 60)
    print("测试 EditFlowTransformer 向后兼容性")
    print("=" * 60)

    from src.modeling.llama_editflow import EditFlowTransformer

    # 创建配置
    class MockConfig:
        def __init__(self):
            self.vocab_size = 50
            self.hidden_dim = 128
            self.num_layers = 2
            self.num_heads = 4
            self.condition_dim = 128
            self.dropout = 0.1
            self.max_seq_len = 24
            self.use_condition_injection = True

    config = MockConfig()

    # 初始化模型
    print(f"\n初始化 EditFlowTransformer...")
    model = EditFlowTransformer(config, verbose=True)

    # 测试前向传播
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    time_steps = torch.rand(batch_size, 1)
    condition = torch.randn(batch_size, 32, config.condition_dim)
    attention_mask = torch.ones(batch_size, seq_len)

    print(f"\n执行前向传播...")
    try:
        with torch.no_grad():
            rates, insert_probs, substitute_probs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                time_steps=time_steps,
                condition=condition
            )

        print(f"✓ 前向传播成功!")
        print(f"\n输出形状:")
        print(f"   rates shape: {rates.shape} (期望: [{batch_size}, {seq_len}, 3])")
        print(f"   insert_probs shape: {insert_probs.shape}")
        print(f"   substitute_probs shape: {substitute_probs.shape}")

        # 验证rates包含三个分量
        ins_rate = rates[..., 0:1]
        del_rate = rates[..., 1:2]
        sub_rate = rates[..., 2:3]

        print(f"\n分离后的速率:")
        print(f"   插入率 shape: {ins_rate.shape}")
        print(f"   删除率 shape: {del_rate.shape}")
        print(f"   替换率 shape: {sub_rate.shape}")

        print(f"\n{'=' * 60}")
        print(f"✓ 向后兼容性测试通过!")
        print(f"{'=' * 60}")

        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("开始测试 LlamaEditFlow 模型\n")

    # 测试1: 基本功能
    test1_passed = test_llama_editflow_basic()

    # 测试2: 向后兼容性
    test2_passed = test_backward_compatibility()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)
    print(f"基本功能测试: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"向后兼容性测试: {'✓ 通过' if test2_passed else '✗ 失败'}")

    if test1_passed and test2_passed:
        print("\n✓ 所有测试通过!")
        sys.exit(0)
    else:
        print("\n✗ 部分测试失败")
        sys.exit(1)
