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

当数据生成卡住时，使用以下命令快速定位问题：
查看日志最后20行

```bash
tail -n 2000 logs/sample_generation.log
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
    --num_samples 10000000\
    --batch_size 48
```

## 分布式训练管理
```bash
pkill -9 train.py
pkill -9 accelerate
```