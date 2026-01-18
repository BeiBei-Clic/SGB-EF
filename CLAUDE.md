# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

LawMind 是一个包含两个独立系统的混合项目：
**MathFixer（知识库）**：一个智能数学公式修正系统，结合了 PostgreSQL 与 GLM（智谱AI/魔塔社区）大语言模型。

### claudecode 运行规则
 - 用中文回答。
 - 使用 uv 管理 Python 虚拟环境，使用该虚拟环境运行 Python 代码。
 - 不要使用 try-except 这种防御性编程，有错误就让它报出来
 - 避免不必要的函数封装，尽可能写简洁。
 - 不要做没有意义、没有信息量的打印和输出。
 - 有关官方代码，尤其是 langchain 相关的，使用 context7 查阅官方文档。
