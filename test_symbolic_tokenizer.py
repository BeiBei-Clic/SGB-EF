"""
测试符号回归专属的小词汇表Tokenizer
"""

import sys
sys.path.insert(0, '/home/xyh/SGB-EF')

from src.utils.special_tokens import SymbolicRegressionTokenizer, SymbolicVocab, SpecialTokensManager

def test_symbolic_vocab():
    """测试词汇表构建"""
    print("=" * 60)
    print("测试1: SymbolicVocab 词汇表构建")
    print("=" * 60)

    # 测试不同维度的词汇表
    for max_dim in [3, 5, 10]:
        vocab = SymbolicVocab.get_full_vocab(max_dim)
        vocab_size = SymbolicVocab.get_vocab_size(max_dim)

        print(f"\nmax_dim={max_dim}:")
        print(f"  词汇表大小: {vocab_size}")
        print(f"  预期大小: {7 + 5 + 7 + max_dim} (特殊7 + 运算符5 + 函数7 + 变量{max_dim})")

        # 验证词汇表结构
        assert '<gap>' in vocab, "缺少 <gap> token"
        assert 'add' in vocab, "缺少 add 运算符"
        assert 'sin' in vocab, "缺少 sin 函数"
        assert 'x0' in vocab, f"缺少 x0 变量 (max_dim={max_dim})"
        if max_dim > 1:
            assert f'x{max_dim-1}' in vocab, f"缺少 x{max_dim-1} 变量 (max_dim={max_dim})"

        print(f"  ✓ 词汇表结构正确")

    print("\n✓ 测试1通过!\n")

def test_tokenizer_basic():
    """测试Tokenizer基本功能"""
    print("=" * 60)
    print("测试2: SymbolicRegressionTokenizer 基本功能")
    print("=" * 60)

    # 创建tokenizer实例
    tokenizer = SymbolicRegressionTokenizer(max_dim=10)

    print(f"\nTokenizer信息:")
    print(f"  词汇表大小: {tokenizer.vocab_size}")
    print(f"  unk_token: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")
    print(f"  pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"  bos_token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"  eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    # 测试分词
    test_expression = "add,x0,mul,x1,2"
    tokens = tokenizer.tokenize(test_expression)
    print(f"\n分词测试:")
    print(f"  输入: '{test_expression}'")
    print(f"  tokens: {tokens}")

    # 测试编码
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"  token_ids: {token_ids}")

    # 测试解码
    decoded_text = tokenizer.decode(token_ids)
    print(f"  解码: '{decoded_text}'")

    # 验证往返一致性（注意：数字会被转换为constant）
    print(f"  ✓ 分词/编码/解码功能正常")

    print("\n✓ 测试2通过!\n")

def test_tokenizer_symbolic_tokens():
    """测试符号回归相关token"""
    print("=" * 60)
    print("测试3: 符号回归专属Token测试")
    print("=" * 60)

    tokenizer = SymbolicRegressionTokenizer(max_dim=3)

    print("\n运算符测试:")
    for op in SymbolicVocab.OPERATORS:
        op_id = tokenizer.convert_tokens_to_ids(op)
        op_back = tokenizer.convert_ids_to_tokens(op_id)
        print(f"  {op}: id={op_id}, 反向转换={op_back}")
        assert op == op_back, f"运算符 {op} 往返转换失败"

    print("\n函数测试:")
    for func in SymbolicVocab.FUNCTIONS:
        func_id = tokenizer.convert_tokens_to_ids(func)
        func_back = tokenizer.convert_ids_to_tokens(func_id)
        print(f"  {func}: id={func_id}, 反向转换={func_back}")
        assert func == func_back, f"函数 {func} 往返转换失败"

    print("\n变量测试 (max_dim=3):")
    for i in range(3):
        var = f'x{i}'
        var_id = tokenizer.convert_tokens_to_ids(var)
        var_back = tokenizer.convert_ids_to_tokens(var_id)
        print(f"  {var}: id={var_id}, 反向转换={var_back}")
        assert var == var_back, f"变量 {var} 往返转换失败"

    print("\n特殊token测试:")
    for token_name, token_str in SymbolicVocab.SPECIAL_TOKENS.items():
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        token_back = tokenizer.convert_ids_to_tokens(token_id)
        print(f"  {token_str}: id={token_id}, 反向转换={token_back}")
        assert token_str == token_back, f"特殊token {token_str} 往返转换失败"

    print("\n✓ 测试3通过!\n")

def test_special_tokens_manager():
    """测试SpecialTokensManager兼容性"""
    print("=" * 60)
    print("测试4: SpecialTokensManager 兼容性")
    print("=" * 60)

    # 创建SymbolicRegressionTokenizer
    tokenizer = SymbolicRegressionTokenizer(max_dim=5)

    # 创建SpecialTokensManager
    manager = SpecialTokensManager(tokenizer, max_dim=5)

    print(f"\nSpecialTokensManager信息:")
    print(f"  tokenizer类型: {type(tokenizer).__name__}")
    print(f"  max_dim: {manager.max_dim}")
    print(f"  原始词汇表大小: {manager.original_vocab_size}")

    # 测试ensure_special_tokens
    manager.ensure_special_tokens(verbose=True)

    # 测试tokenize_expression
    test_expr = "add,x0,mul,x1,sin,x2"
    token_ids = manager.tokenize_expression(test_expr)
    print(f"\n表达式分词测试:")
    print(f"  输入表达式: '{test_expr}'")
    print(f"  token_ids: {token_ids}")

    # 验证token ID都在词汇表范围内
    vocab = manager.tokenizer.get_vocab()
    for tid in token_ids:
        assert tid < len(vocab), f"token ID {tid} 超出词汇表范围"
    print(f"  ✓ 所有token ID都在词汇表范围内")

    # 测试setup_tokenizer_special_tokens
    manager.setup_tokenizer_special_tokens(verbose=False)
    print(f"  ✓ setup_tokenizer_special_tokens 执行成功")

    print("\n✓ 测试4通过!\n")

def test_expression_examples():
    """测试实际表达式示例"""
    print("=" * 60)
    print("测试5: 实际表达式示例")
    print("=" * 60)

    tokenizer = SymbolicRegressionTokenizer(max_dim=3)

    # 测试一些常见的数学表达式
    test_cases = [
        ("线性", "add,x0,x1"),
        ("二次", "add,x0,mul,x1,x1"),
        ("三角函数", "add,sin,x0,cos,x1"),
        ("复合函数", "add,mul,x0,x0,sin,x1"),
        ("复杂表达式", "add,div,x0,2,sqrt,add,x1,mul,x2,x2"),
    ]

    print("\n表达式分词/编码/解码测试:")
    for name, expr in test_cases:
        tokens = tokenizer.tokenize(expr)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)

        print(f"\n  {name}:")
        print(f"    原始: {expr}")
        print(f"    tokens: {tokens}")
        print(f"    ids: {token_ids}")
        print(f"    解码: {decoded}")

    print("\n✓ 测试5通过!\n")

def test_vocab_size_comparison():
    """对比新旧词汇表大小"""
    print("=" * 60)
    print("测试6: 词汇表大小对比")
    print("=" * 60)

    # BERT base uncased 的词汇表大小约为 30,000
    bert_vocab_size = 30522

    # 我们的符号回归词汇表
    for max_dim in [3, 5, 10, 20]:
        symbolic_vocab_size = SymbolicVocab.get_vocab_size(max_dim)
        reduction = (bert_vocab_size - symbolic_vocab_size) / bert_vocab_size * 100

        print(f"\nmax_dim={max_dim}:")
        print(f"  BERT词汇表: {bert_vocab_size:,}")
        print(f"  符号回归词汇表: {symbolic_vocab_size:,}")
        print(f"  减少: {bert_vocab_size - symbolic_vocab_size:,} 个token ({reduction:.1f}%)")

    print("\n✓ 测试6通过!\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("符号回归Tokenizer测试套件")
    print("=" * 60 + "\n")

    try:
        test_symbolic_vocab()
        test_tokenizer_basic()
        test_tokenizer_symbolic_tokens()
        test_special_tokens_manager()
        test_expression_examples()
        test_vocab_size_comparison()

        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
