"""
测试LLaMA EditFlow与EditFlowManager的集成
"""
import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.modeling.llama_editflow import EditFlowTransformer, LlamaEditFlowConfig
from src.modeling.condition_encoder import ConditionEncoder


def test_model_initialization():
    """测试模型初始化是否正常"""
    print("=" * 60)
    print("测试 1: 模型初始化")
    print("=" * 60)

    # 模拟args对象
    class Args:
        def __init__(self):
            self.condition_dim_hidden = 128
            self.condition_num_seeds = 32
            self.hidden_dim = 256
            self.n_layers = 4
            self.n_heads = 8
            self.dropout = 0.1
            self.max_expr_length = 24
            self.use_condition_injection = True
            # 条件编码器参数
            self.condition_max_input_dim = 6
            self.condition_dim_hidden = 128
            self.condition_num_heads = 4
            self.condition_num_inds = 32
            self.condition_num_layers = 3
            self.condition_num_seeds = 32
            self.condition_use_sinusoidal_encoding = True
            self.condition_sinusoidal_dim = 64
            self.condition_sinusoidal_max_freq = 10000.0

    args = Args()

    # 创建配置
    config = LlamaEditFlowConfig(
        vocab_size=29,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        condition_dim=args.condition_dim_hidden,
        dropout=args.dropout,
        max_seq_len=args.max_expr_length,
        use_condition_injection=args.use_condition_injection,
    )

    print(f"✓ 配置创建成功")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")

    # 创建模型
    model = EditFlowTransformer(config, verbose=True)

    print(f"✓ 模型初始化成功")
    print(f"  模型类型: {type(model).__name__}")
    print(f"  有config属性: {hasattr(model, 'config')}")
    print(f"  config类型: {type(model.config).__name__}")

    # 创建条件编码器
    condition_encoder = ConditionEncoder(
        model_name=None,  # 不使用预训练模型
        verbose=False,
        max_length=None,
        args=args
    )

    print(f"✓ 条件编码器初始化成功")

    return model, condition_encoder, config


def test_forward_pass(model, condition_encoder, config):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试 2: 前向传播")
    print("=" * 60)

    device = torch.device('cpu')  # 使用CPU测试

    # 准备测试数据
    batch_size = 2
    seq_len = 10
    n_points = 50

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    time_steps = torch.rand(batch_size, 1)
    attention_mask = torch.ones(batch_size, seq_len)

    # 准备条件编码器输入
    x_values = torch.randn(batch_size, n_points, 3)  # 3维输入
    residuals = torch.randn(batch_size, n_points)

    print(f"输入准备:")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  time_steps shape: {time_steps.shape}")
    print(f"  x_values shape: {x_values.shape}")
    print(f"  residuals shape: {residuals.shape}")

    # 条件编码
    with torch.no_grad():
        condition = condition_encoder(x_values, residuals)
        print(f"\n条件编码:")
        print(f"  condition shape: {condition.shape}")

        # 模型前向传播
        rates, insert_probs, substitute_probs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            time_steps=time_steps,
            condition=condition
        )

    print(f"\n✓ 前向传播成功")
    print(f"输出:")
    print(f"  rates shape: {rates.shape} (期望: [{batch_size}, {seq_len}, 3])")
    print(f"  insert_probs shape: {insert_probs.shape} (期望: [{batch_size}, {seq_len}, {config.vocab_size}])")
    print(f"  substitute_probs shape: {substitute_probs.shape} (期望: [{batch_size}, {seq_len}, {config.vocab_size}])")

    # 验证输出形状
    assert rates.shape == (batch_size, seq_len, 3), f"rates形状错误: {rates.shape}"
    assert insert_probs.shape == (batch_size, seq_len, config.vocab_size), f"insert_probs形状错误: {insert_probs.shape}"
    assert substitute_probs.shape == (batch_size, seq_len, config.vocab_size), f"substitute_probs形状错误: {substitute_probs.shape}"

    # 验证概率和为1
    prob_sum = insert_probs[0, 0].sum().item()
    assert abs(prob_sum - 1.0) < 1e-5, f"概率和不为1: {prob_sum}"

    print(f"\n✓ 所有验证通过")

    return True


def test_config_access():
    """测试config属性访问"""
    print("\n" + "=" * 60)
    print("测试 3: Config属性访问")
    print("=" * 60)

    config = LlamaEditFlowConfig(
        vocab_size=50,
        hidden_dim=256,
        n_layers=6,
        n_heads=8,
    )

    print(f"测试config属性访问:")
    print(f"  config.vocab_size: {config.vocab_size}")
    print(f"  config.hidden_dim: {config.hidden_dim}")
    print(f"  config.n_layers: {config.n_layers}")
    print(f"  config.n_heads: {config.n_heads}")

    # 创建模型并检查config
    model = EditFlowTransformer(config, verbose=False)

    print(f"\n模型config:")
    print(f"  model.config: {model.config}")
    print(f"  model.config.vocab_size: {model.config.vocab_size}")
    print(f"  model.config.__class__.__name__: {model.config.__class__.__name__}")

    # 检查是否有vocab_size属性
    assert hasattr(model.config, 'vocab_size'), "model.config没有vocab_size属性"
    assert model.config.vocab_size == 50, f"vocab_size值错误: {model.config.vocab_size}"

    print(f"\n✓ Config访问测试通过")
    return True


def main():
    print("开始LLaMA EditFlow集成测试\n")

    try:
        # 测试1: 模型初始化
        model, condition_encoder, config = test_model_initialization()

        # 测试2: 前向传播
        test_forward_pass(model, condition_encoder, config)

        # 测试3: Config访问
        test_config_access()

        print("\n" + "=" * 60)
        print("✓ 所有集成测试通过!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
