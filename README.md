# 环境配置

## Hugging Face 镜像源

为了加速模型下载，建议使用国内镜像源。

### 设置方法

```bash
# 临时设置
export HF_ENDPOINT=https://hf-mirror.com
```

## 模型缓存

### 缓存位置

项目模型缓存默认存储在：`models/huggingface_cache/`

### 嵌入模型对比

| 模型名称 | 参数量 | 嵌入维度 | 特点 | 适用场景 |
|---------|--------|----------|------|----------|
| **nomic-ai/nomic-embed-text-v1.5** | 137M | 768维 | ✅ 轻量级<br>✅ 推理快速<br>✅ 开源Apache 2.0 | 资源受限环境<br>快速原型验证<br>生产环境部署 |
| **Qwen/Qwen3-Embedding-0.6B** | 600M | - | 🔥 高性能<br>🔥 参数量大<br>🔥 可能更好效果 | 追求最佳效果<br>充足计算资源<br>研究实验 |

### 查看日志
```bash
tail -n 500 logs/sample_generation.log

tail -n 500 logs/training_debug.log

tail -n 500 logs/training.log

tail -n 500 logs/inference.log
```

### 调试日志内容
`training_debug.log` 记录了每个batch的详细信息：
- 输入张量统计（x_values, residuals, condition_embeddings）
- 模型输出统计（pred_rates, pred_ins_probs, pred_sub_probs）
- 损失值（LOSS_COMPUTED）
- 梯度统计（GRAD_STATS: max, min, has_nan）
- 梯度范数（GRAD_NORM）
- NaN错误信息（ERROR日志）


## 分布式训练管理
```bash
pkill -9 train.py
pkill -9 accelerate
```

## 数据生成时解决卡死的两个关键
- **样本级别超时保护**：使用with_timeout装饰器，为每个样本设置超时时间，防止单个样本生成时间过长导致整个批次卡死。
- **表达式生成时递归深度限制**：为了防止无限递归，设置最大递归深度，超过该深度时抛出异常。

## 监控GPU和CPU使用情况
```bash
# 查看GPU占用情况
watch -n 1 -d nvidia-smi
# 查看进程占用的CPU和内存
watch -n 2 'ps aux | grep "train.py" | grep -v grep | awk "{print \$3,\$2}" | head -3'
# 查看文件是否在被访问
lsof | grep "flow_samples_10000000"
```

## tmux 将终端任务挂到后台运行
```bash
# 创建一个名为my_session的tmux会话
tmux new -s my_session
# 在tmux会话中运行任务
accelerate launch \
    --num_processes=3 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --multi_gpu \
    train.py \
    --num_epochs 50 \
    --num_samples 10000000 \
    --batch_size 960 \
    --dataset_stream False

accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    train.py \
    --num_epochs 50 \
    --num_samples 1\
    --batch_size 32 \
    --debug 1

# 将会话挂到后台
Ctrl + B, D
# 查看当前有哪些正在运行的会话
tmux ls
# 进入（恢复）指定名字的会话
tmux a -t my_session
#在tmux内部关闭会话
Ctrl + D
# 在tmux外部强制关闭会话
tmux kill-session -t my_session
# 进入复制模式
Ctrl + B, [
# 选择要复制的内容
# 复制到剪贴板
Ctrl + C
# 退出复制模式
Ctrl + D
```