# 环境配置

## Hugging Face 镜像源

为了加速模型下载，建议使用国内镜像源。

### 设置方法

```bash
# 临时设置
export HF_ENDPOINT=https://hf-mirror.com

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 验证设置

```bash
echo $HF_ENDPOINT
# 输出: https://hf-mirror.com
```

### 可用镜像源

- **主镜像**: `https://hf-mirror.com`
- **备用镜像**: `https://hf.1zhe.icu`

## 模型缓存

### 缓存位置

项目模型缓存默认存储在：`models/huggingface_cache/`

### 嵌入模型对比

| 模型名称 | 参数量 | 嵌入维度 | 特点 | 适用场景 |
|---------|--------|----------|------|----------|
| **nomic-ai/nomic-embed-text-v1.5** | 137M | 768维 | ✅ 轻量级<br>✅ 推理快速<br>✅ 开源Apache 2.0 | 资源受限环境<br>快速原型验证<br>生产环境部署 |
| **Qwen/Qwen3-Embedding-0.6B** | 600M | - | 🔥 高性能<br>🔥 参数量大<br>🔥 可能更好效果 | 追求最佳效果<br>充足计算资源<br>研究实验 |

## 数据生成日志监控

### 日志文件

- **详细日志**: `logs/sample_generation.log` - 记录所有样本生成步骤
- **卡住样本日志**: `logs/sample_stuck.log` - 只记录出错的样本信息

### 实时监控

```bash
# 查看正在生成的样本
tail -f logs/sample_generation.log

# 查看卡住的样本
tail -f logs/sample_stuck.log

# 查看最新数据文件
ls -la data/ | tail -5

# 查看数据生成进度
ls -la data/*batch*.txt | wc -l
```

### 日志分析

```bash
# 统计卡住样本数量
grep -c "卡住样本记录" logs/sample_stuck.log

# 查看最常见的错误类型
grep "错误:" logs/sample_stuck.log | sort | uniq -c | sort -nr

# 查看数据生成时间分布
grep "开始生成" logs/sample_generation.log | awk '{print $1}' | sort | uniq -c

# 查看当前批次进度
grep "第.*批" logs/sample_generation.log | tail -1

# 统计复杂表达式样本
grep "跳过复杂表达式" logs/sample_generation.log | wc -l
```

## 分布式训练

```bash
accelerate launch \
    --num_processes=3 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=no \
    --multi_gpu \
    train.py \
    --num_samples 1000\
    --batch_size 48
```

## 分布式训练管理
```bash
pkill -9 train.py
pkill -9 accelerate
```