# MoE Gated DeltaNet (MoE-GDN)

本分支在 Megatron-LM 中实现了 **MoE Gated DeltaNet**——一种将 Mixture-of-Experts 路由机制引入 Gated Delta Rule 线性注意力的架构变体。核心思想是将线性注意力的多个 head 视为 "专家"，通过 top-k 路由选择性地激活部分 head 进行状态写入和读取，从而在保持线性复杂度的同时大幅扩展模型的状态容量。

## 架构概览

MoE-GDN 层替换 Transformer 中的标准注意力层，其核心设计参考: [Gated Delta Net 及 MoE Gated Delta Net 稀疏化方案](https://zhuanlan.zhihu.com/p/2025182354461696359)


### 关键设计

1. **Shared Heads + Routed Heads**：总 head 数 = `num_shared_heads` + `num_routed_heads`。Shared heads 始终激活（权重恒为 1），Routed heads 通过 top-k 路由按需激活。
2. **读写解耦路由**：Write Router 和 Read Router 分别独立路由，支持不同的 top-k 值（`write_topk` / `read_topk`）。读取分数融合了写入分数：`read_score = w * write_score + (1-w) * read_sigmoid`，其中 `w = linear_write_coeff_for_read`，让模型适当更加关注当前写入的状态，避免只从历史信息中提取信息，也要从当前更新的即时信息中提取信息。
3. **路由写入应该根据写入路由系数考虑写入的”量级”**：将写入系数 alpha，beta 区分开，只对写入系数使用路由系数加权，而不对遗忘系数进行加权，避免长期运行导致状态的尺度急剧缩小，甚至”消失”
3. **Loss-Free Load Balancing**：可选启用 Expert Bias（`linear_moe_router_enable_expert_bias`），在每个 optimizer step 后基于 token 分配统计更新 bias，无需额外 auxiliary loss。
4. **Gated Delta Rule**：核心状态更新使用 FLA 库的 `chunk_gated_delta_rule` 高效 kernel，同时提供纯 PyTorch 的确定性回退实现。
5. **Per-Head Zero-Centered RMSNorm**：输出归一化采用减去均值后再 RMS 归一化的方式，按 head 独立处理。

## 配置参数

在 `TransformerConfig` 中设置以下参数启用 MoE-GDN：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `experimental_attention_variant` | `None` | 设为 `"moe_gated_delta_net"` 启用 |
| `linear_attention_freq` | `None` | 线性注意力层的频率，如 `4` 表示每 4 层中 3 层为线性注意力、1 层为标准 SDPA |
| `linear_key_head_dim` | `128` | Q/K 的 head 维度 |
| `linear_value_head_dim` | `128` | V/Gate 的 head 维度 |
| `linear_num_shared_heads` | `None` | 始终激活的 shared head 数量 |
| `linear_num_routed_heads` | `None` | 路由专家池的 head 数量 |
| `linear_write_topk` | `None` | 状态写入时选择的 top-k head 数 |
| `linear_read_topk` | `None` | 状态读取时选择的 top-k head 数 |
| `linear_write_coeff_for_read` | `0.5` | 读取分数中写入分数的融合系数 |
| `linear_moe_router_enable_expert_bias` | `False` | 是否启用 loss-free load balancing 的 expert bias |

### 约束条件

- `write_topk <= num_routed_heads`，`read_topk <= num_routed_heads`
- `(num_shared_heads + num_routed_heads) % tensor_model_parallel_size == 0`
- `num_routed_heads % tensor_model_parallel_size == 0`
- 当前 **不支持** Context Parallelism（`context_parallel_size` 必须为 1）

## 文件结构

```
megatron/core/ssm/moe_gated_delta_net.py          # 核心模块实现
megatron/core/transformer/transformer_config.py    # 配置定义与校验
megatron/core/models/gpt/
  experimental_attention_variant_module_specs.py    # ModuleSpec 构建与层模式分配
megatron/core/distributed/finalize_model_grads.py  # Expert bias 梯度后更新
megatron/training/training.py                      # FLOPs 计算适配
megatron/training/arguments.py                     # CLI 参数定义
examples/gdn/pretrain_qwen35_moe_gdn_2b.sh        # 训练脚本示例
```

## 快速开始

参考 `examples/gdn/pretrain_qwen35_moe_gdn_2b.sh`，关键参数：

```bash
LINEAR_ATTN_ARGS=(
    --experimental-attention-variant moe_gated_delta_net
    --linear-attention-freq 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-shared-heads 4
    --linear-num-routed-heads 64
    --linear-write-topk 8
    --linear-read-topk 16
    --linear-write-coeff-for-read 0.5
    --linear-moe-router-enable-expert-bias
)
```

### 依赖

- **flash-linear-attention (FLA)**：`pip install flash-linear-attention`，提供高效的 `chunk_gated_delta_rule` kernel

## 待优化项 (TODOs)

1. **QKV EMA Smooth 未启用**
   - `smooth_qkv` 方法已实现，利用 `A_log` 和 `dt_bias` 参数通过关联扫描进行可学习的 EMA 平滑
   - 但目前的实现在反向传播的时候没有使用重计算方式，在序列维度会累积巨大的计算图，导致显存爆炸，需要进一步优化

2. **Masking 策略优化**
   - 当前使用乘法 masking（`query * read_mask`）方案通过将对应的专家输入置为零，实现未激活专家的计算，后续进一步优化需要完全避免未激活专家的计算，包括 gated_delta_rule 内部对于未激活的状态跳过写入或读取
   - 对于未激活的 head，当前仍然会参与 chunk_gated_delta_rule 的计算（只是输入为零），理想情况下应完全跳过，这部分需要重写 kernel 实现

3. **序列维度路由负载均衡**
	- 全局路由负载均衡避免专家整体不被选中，序列路由负载均避免序列维度上长时间不更新或不被读取，成为”死专家”，或者长时间不更新，但突然更新，破坏状态，这一点跟 MLP 还不太一样，MLP 权重是静态的，不会随着序列修改，所以这里必须考虑序列维度的路由负载均衡，目前尚未实现，在验证架构有效之后再进一步优化

