# G0.5 Documentation Index

## Quick Start

| Document | Contents |
|----------|----------|
| [QUICK_START.md](../configs/QUICK_START.md) | Config cheat sheet: where to change common settings, validation commands, and how to add your own dataset. |

## Architecture

| Document | Contents |
|----------|----------|
| [config.md](architecture/config.md) | Hydra four-layer composition, task inheritance, and mixture configuration. |
| [tokenizer.md](architecture/tokenizer.md) | Action tokenizer four-layer architecture, backend families, and embodiment-specific action spaces. |
| [parts_meta.md](architecture/parts_meta.md) | Full lifecycle of `parts_meta`: YAML structure, variant comparison, merger families, and checklist for adding new metadata. |

## G05 Model

- [G05 Architecture Design](architecture/g05_architecture.md): core architecture, homogeneous Mixture design, Policy/Model/Helper layering, precision policy, and end-to-end data flow.
- [G05 Config Design](architecture/g05_config.md): dual config modes, HF auto vs YAML override, Mixture initialization entries, G05Model assembly, and checkpoint weight mapping.
- [G05 I/O Format](architecture/g05_io.md): full Dataset-to-Model I/O path, exact component shapes, processor APIs, attention-mask `TOKEN_INDEX` encoding, and KV-cache communication across Mixtures.
- [Qwen3.5 Design](architecture/qwen35_design.md): Qwen3.5 backbone differences, including MRoPE, GatedDeltaNet, SparseKVCache, and Conv3D Vision.
- [Qwen3.5 MEM Overview](architecture/qwen_mem_overview.md): multi-frame memory mechanism and data flow.

## Data

| Document | Contents |
|----------|----------|
| [schema.md](data/schema.md) | `shape_meta` definition and steps for adding a new embodiment. |
| [samples_builders.md](data/samples_builders.md) | SamplesBuilder catalog: field dependencies, templates, decision flowchart, and config examples. |

## Deployment

| Document | Contents |
|----------|----------|
| [inference.md](deployment/inference.md) | Real-robot inference flow for R1LITE. |
| [serve_policy.md](deployment/serve_policy.md) | WebSocket inference service: startup, dynamic batching, client protocol, and torch.compile acceleration. |
| [serve_policy_mem.md](deployment/serve_policy_mem.md) | Multi-frame memory inference service. |
| [model_debug_logging.md](deployment/model_debug_logging.md) | Model debug logging system for torch.compile and distributed environments. |
