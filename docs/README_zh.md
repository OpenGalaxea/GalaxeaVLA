# G0.5 文档索引

## 快速上手

| 文档 | 内容 |
|------|------|
| [QUICK_START.md](../configs/QUICK_START.md) | Config 速查：想改 X → 改哪个文件、验证命令、接入自己的数据集 |

## 架构

| 文档 | 内容 |
|------|------|
| [config.md](architecture/config.md) | Hydra 4 层组合、Task 继承体系、Mixture 配置 |
| [tokenizer.md](architecture/tokenizer.md) | Action Tokenizer 4 层架构、后端家族、按 embodiment 配置的动作空间 |
| [parts_meta.md](architecture/parts_meta.md) | parts_meta 全生命周期：yaml 结构、变体对比、Merger 家族、新增 checklist |

## G05 模型

- [G05 架构设计](architecture/g05_architecture.md) — 核心架构：Mixture 同构设计、层次关系（Policy/Model/Helper）、精度策略、端到端数据流
- [G05 Config 设计](architecture/g05_config.md) — Config 双模式（HF auto / YAML override）、Mixture 初始化入口、G05Model 组装、checkpoint 权重映射
- [G05 I/O 格式](architecture/g05_io.md) — 全链路 I/O（Dataset → Model）：各组件精确 shape、Processor 三 API、Attention Mask TOKEN_INDEX 编码、KV Cache 跨 Mixture 通信
- [Qwen3.5 设计文档](architecture/qwen35_design.md) — Qwen3.5 backbone 架构差异（MRoPE、GatedDeltaNet、SparseKVCache、Conv3D Vision）
- [Qwen3.5 MEM 多帧概述](architecture/qwen_mem_overview.md) — 多帧记忆（MEM）机制与数据流

## 数据

| 文档 | 内容 |
|------|------|
| [schema.md](data/schema.md) | shape_meta 定义、新增 embodiment 步骤 |
| [samples_builders.md](data/samples_builders.md) | SamplesBuilder 全家福：字段依赖、模板、决策流程图、Config 示例 |

## 部署

| 文档 | 内容 |
|------|------|
| [inference.md](deployment/inference.md) | 真机推理流程（R1LITE） |
| [serve_policy.md](deployment/serve_policy.md) | WebSocket 推理服务：启动、动态批处理、Client 协议、torch.compile 加速 |
| [serve_policy_mem.md](deployment/serve_policy_mem.md) | 多帧记忆（MEM）推理服务 |
| [model_debug_logging.md](deployment/model_debug_logging.md) | torch.compile / 分布式环境下的模型调试日志系统 |
