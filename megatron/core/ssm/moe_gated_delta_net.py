# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear, get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

# TODO: Implement GatedDeltaNetContextParallel
# from .gated_delta_net_context_parallel import GatedDeltaNetContextParallel

try:
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    chunk_gated_delta_rule = None

    HAVE_FLA = False

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None


logger = logging.getLogger(__name__)


class PerHeadZeroCenteredRMSNorm(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6
    ):
        """RMS Normaliation module

        Args:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))
        setattr(self.weight, "tensor_model_parallel", True)
        setattr(self.weight, "partition_dim", 0)

    def _norm(self, x):
        # x: (..., num_heads, head_dim)
        x = x - x.mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@dataclass
class MoEGatedDeltaNetSubmodules:
    """
    Contains the module specs for MoEGatedDeltaNet:
    in_proj, out_proj, writer_router, read_router.
    """
    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp
    writer_router: Union[ModuleSpec, type] = IdentityOp
    read_router: Union[ModuleSpec, type] = IdentityOp


class MoEGatedDeltaNet(MegatronModule):
    """Gated Delta Net (GDN) layer class

    GDN layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MoEGatedDeltaNetSubmodules,
        layer_number: int = None,
        bias: bool = False,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
    ):
        """
        Args:
            config: The config of the model.
            submodules: Contains the module specs for the input and output linear layers.
            layer_number: The layer number of this GDN layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization in the kernel of the gated delta rule.
            A_init_range: The initialization range for the attention weights.
            pg_collection: The required process groups to use for tensor model parallel and context
                parallel.
        """

        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install it with `pip install flash-linear-attention`."
            )

        super().__init__(config)

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet"
        self.pg_collection = pg_collection
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        # self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.query_key_dim = config.linear_key_head_dim
        self.value_dim = config.linear_value_head_dim
        self.num_shared_heads = config.linear_num_shared_heads # TODO
        self.num_routed_heads = config.linear_num_routed_heads # TODO
        self.num_total_heads = self.num_shared_heads + self.num_routed_heads
        self.num_heads_local_tp = self.num_total_heads // self.tp_size
        self.qk_dim = self.query_key_dim * self.num_total_heads
        self.v_dim = self.value_dim * self.num_total_heads
        self.write_topk = config.linear_write_topk
        self.read_topk = config.linear_read_topk
        self.write_coeff_for_read = getattr(config, 'linear_write_coeff_for_read', 0.5)

        # self.write_router = build_module(
        #     submodules.writer_router,
        #     self.hidden_size,
        #     self.num_routed_heads,
        #     config=self.config,
        #     init_method=self.config.init_method,
        #     gather_output=True,
        #     bias=bias,
        #     skip_bias_add=False,
        #     is_expert=False,
        #     tp_comm_buffer_name="writer_router",
        #     tp_group=self.pg_collection.tp,
        # )
        # self.read_router = build_module(
        #     submodules.read_router,
        #     self.hidden_size,
        #     self.num_routed_heads,
        #     config=self.config,
        #     init_method=self.config.init_method,
        #     gather_output=True,
        #     bias=bias,
        #     skip_bias_add=False,
        #     is_expert=False,
        #     tp_comm_buffer_name="read_router",
        #     tp_group=self.pg_collection.tp,
        # )

        self.write_router = ColumnParallelLinear(
            self.hidden_size,
            self.num_routed_heads,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=True,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="writer_router",
            tp_group=self.pg_collection.tp,
        )
        self.read_router = ColumnParallelLinear(
            self.hidden_size,
            self.num_routed_heads,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=True,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="read_router",
            tp_group=self.pg_collection.tp,
        )

        self.enable_expert_bias = getattr(config, 'linear_moe_router_enable_expert_bias', False)
        self.expert_bias_update_rate = getattr(config, 'moe_router_bias_update_rate', 1e-3)

        if self.enable_expert_bias:
            # Persistent: saved to checkpoint
            self.register_buffer(
                'write_expert_bias',
                torch.zeros(self.num_routed_heads, dtype=torch.float32,
                            device=torch.cuda.current_device()),
            )
            self.register_buffer(
                'read_expert_bias',
                torch.zeros(self.num_routed_heads, dtype=torch.float32,
                            device=torch.cuda.current_device()),
            )
            # Non-persistent: accumulates token counts per step, not checkpointed
            # Shape: (num_routed_heads // tp_size,) - only routed heads, local TP slice
            self.register_buffer(
                'write_local_tokens_per_head',
                torch.zeros(self.num_routed_heads // self.tp_size, dtype=torch.float32,
                            device=torch.cuda.current_device()),
                persistent=False,
            )
            self.register_buffer(
                'read_local_tokens_per_head',
                torch.zeros(self.num_routed_heads // self.tp_size, dtype=torch.float32,
                            device=torch.cuda.current_device()),
                persistent=False,
            )
        else:
            self.write_expert_bias = None
            self.read_expert_bias = None
            self.write_local_tokens_per_head = None
            self.read_local_tokens_per_head = None

        # Input projection (hidden_states -> q, k, v, gate, beta, alpha)
        # TODO: for now, output gate is forced for GDN.
        # We may remove this restriction in the future.
        # query_dim * n_heads + key_dim * n_heads + value_dim * n_heads 
        # + gate_dim(=value_dim) * n_heads + 1(=beta) * n_heads + 1(=alpha) * n_heads
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_total_heads * 2
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN layer "
                "input projection output tensor must be a multiple of 16."
            )
        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )


        # Time step projection (discretization)
        # A_log and dt_bias 除了用于 alpha 计算外，还用于计算 qkv 的 EMA 可学习的平滑系数
        # dt_bias parameter
        self.dt_bias = nn.Parameter(
            torch.empty(
                (self.num_heads_local_tp, 4),
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)
        # A_log parameter
        self.A_log = nn.Parameter(
            torch.empty(
                (self.num_heads_local_tp, 4),
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        setattr(self.A_log, "partition_dim", 0)

        # # Output layernorm before projection
        # self.out_norm = build_module(
        #     submodules.out_norm,
        #     config=self.config,
        #     hidden_size=self.value_head_dim,
        #     eps=self.config.layernorm_epsilon,
        # )
        self.out_norm = PerHeadZeroCenteredRMSNorm(self.num_heads_local_tp, self.value_dim)

        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
        )

        # TODO: support CP

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                # dt_bias
                torch.ones(
                    (self.num_heads_local_tp, 4),
                    out=self.dt_bias.data,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                )
                # A_log
                A = torch.empty(
                    (self.num_heads_local_tp, 4),
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(torch.log(A))

    def forward_local_write_read_gate(self, hidden_states):
        """
        计算当前 TP Rank 上专家的写入和读取路由分数以及 mask
        Args:
            hidden_states: [bsz, seq_len, hidden_size]
        Return:
            write_score: [bsz, seq_len, local_heads]
            write_mask: [bsz, seq_len, local_heads]
            read_score: [bsz, seq_len, local_heads]
            read_mask: [bsz, seq_len, local_heads]
        """
        bsz, seq_len, _ = hidden_states.shape
        # Write & Read Router
        nvtx_range_push(suffix="write_router")
        write_score, _ = self.write_router(hidden_states)  # [bsz, seq_len, num_routed_heads]
        nvtx_range_pop(suffix="write_router")

        nvtx_range_push(suffix="read_router")
        read_score, _ = self.read_router(hidden_states)    # [bsz, seq_len, num_routed_heads]
        nvtx_range_pop(suffix="read_router")

        write_scores = write_score.float().sigmoid()  # [bsz, seq_len, num_routed_heads], in [0,1]
        read_scores  = self.write_coeff_for_read * write_scores + (1 - self.write_coeff_for_read) * read_score.float().sigmoid()   # [bsz, seq_len, num_routed_heads], in [0,1]

        if self.enable_expert_bias:
            write_scores_for_routing = write_scores + self.write_expert_bias
            read_scores_for_routing  = read_scores + self.read_expert_bias
        else:
            write_scores_for_routing = write_scores
            read_scores_for_routing  = read_scores

        # Top-k selection on biased scores (descending)
        _, write_topk_indices = torch.topk(
            write_scores_for_routing, self.write_topk, dim=-1, largest=True, sorted=True
        )  # [bsz, seq_len, write_topk]
        # Weights: gather from UNBIASED sigmoid scores
        write_weight = torch.gather(write_scores, dim=-1, index=write_topk_indices)

        _, read_topk_indices = torch.topk(
            read_scores_for_routing, self.read_topk, dim=-1, largest=True, sorted=True
        )  # [bsz, seq_len, read_topk]
        read_weight = torch.gather(read_scores, dim=-1, index=read_topk_indices)

        # Boolean selection masks: [s, b, num_routed_heads]
        write_mask = torch.zeros_like(write_scores, dtype=torch.bool)
        write_mask.scatter_(-1, write_topk_indices, True)

        read_mask = torch.zeros_like(read_scores, dtype=torch.bool)
        read_mask.scatter_(-1, read_topk_indices, True)

        
        
        # ── Extract current TP rank's local routing scores and masks ──────────
        # ColumnParallelLinear (gather_output=False) assigns heads contiguously:
        # rank r owns global head indices [r*H, (r+1)*H), H = num_routed_heads // tp_size
        tp_rank = torch.distributed.get_rank(group=self.pg_collection.tp)
        heads_per_rank = self.num_routed_heads // self.tp_size
        local_start = tp_rank * heads_per_rank
        local_end   = local_start + heads_per_rank

        # Local boolean masks: [bsz, seq_len, local_heads]
        write_mask_local = write_mask[..., local_start:local_end]
        read_mask_local  = read_mask[..., local_start:local_end]

        # Local routing weights (unbiased sigmoid, zeroed for non-selected heads)
        # Shape: [b, s, heads_per_rank]
        write_weight_local = (
            write_scores[..., local_start:local_end].type_as(hidden_states)
            * write_mask_local
        )
        read_weight_local = (
            read_scores[..., local_start:local_end].type_as(hidden_states)
            * read_mask_local
        )

        # Accumulate token counts for bias update (skip during eval / activation recompute)
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.write_local_tokens_per_head += (
                    write_mask_local.reshape(-1, heads_per_rank).float().sum(dim=0)
                )
                self.read_local_tokens_per_head += (
                    read_mask_local.reshape(-1, heads_per_rank).float().sum(dim=0)
                )

        # cat shared expert scores
        local_shared_heads = self.num_shared_heads // self.tp_size
        local_shared_weights = torch.ones((bsz, seq_len, local_shared_heads), device=write_weight_local.device, dtype=write_weight_local.dtype)
        local_shared_masks = torch.ones((bsz, seq_len, local_shared_heads), device=read_mask_local.device, dtype=read_mask_local.dtype)

        write_weight_local = torch.cat([local_shared_weights, write_weight_local], dim=-1)
        read_weight_local = torch.cat([local_shared_weights, read_weight_local], dim=-1)
        write_mask_local = torch.cat([local_shared_masks, write_mask_local], dim=-1)
        read_mask_local = torch.cat([local_shared_masks, read_mask_local], dim=-1)

        return write_weight_local, write_mask_local, read_weight_local, read_mask_local

    def forward_qkv_gate_alpha_beta(self, hidden_states):
        """
        计算当前 TP Rank 上专家的写入和读取路由分数以及 mask
        Args:
            hidden_states: [bsz, seq_len, hidden_size]
        Return:
            qkv: [bsz, seq_len, local_heads, qkv_dim]
            gate: [bsz, seq_len, local_heads, value_dim]
            alpha: [bsz, seq_len, local_heads]
            beta: [bsz, seq_len, local_heads]
        """
        # Input projection
        bsz, seq_len, _ = hidden_states.shape
        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        local_heads = self.num_total_heads // self.tp_size
        # Split, reorder, and reshape the tensor into q, k, v, gate, beta, alpha
        query, key, value, gate, beta, alpha = torch.split(
            qkvzba,
            [
                # (self.qk_dim * 2 + self.v_dim) // self.tp_size,
                self.qk_dim // self.tp_size,
                self.qk_dim // self.tp_size,
                self.v_dim // self.tp_size,
                self.v_dim // self.tp_size,
                local_heads,
                local_heads,
            ],
            dim=-1,
        )
        query = query.reshape(bsz, seq_len, local_heads, self.query_key_dim)
        key = key.reshape(bsz, seq_len, local_heads, self.query_key_dim)
        value = value.reshape(bsz, seq_len, local_heads, self.value_dim)
        gate = gate.reshape(bsz, seq_len, local_heads, self.value_dim)
        beta = beta.reshape(bsz, seq_len, local_heads)
        alpha = alpha.reshape(bsz, seq_len, local_heads)

        return query, key, value, gate, alpha, beta


    def compute_local_factors(self, alpha, beta):
        """
        Args:
            alpha: [bsz, seq_len, local_heads]
            beta: [bsz, seq_len, local_heads]
            A_log: [local_heads, 4]
        Return:
            alpha: [bsz, seq_len, local_heads] use A, dt
            beta: [bsz, seq_len, local_heads]
            qkv_smooth_factor: [local_heads, 3] use A, dt in (0, 1)
        """
        bsz, seq_len, local_heads = alpha.shape
        A = self.A_log.float().exp()
        b = self.dt_bias.float()
        alpha_log = -A[:, 0] * F.softplus(alpha.float() + b[:, 0])
        beta = beta.sigmoid()
        qkv_smooth_factor = torch.exp(-A[:, 1:] * F.softplus(b[:, 1:]))
        return alpha_log, beta, qkv_smooth_factor


    def smooth_qkv(self, query, key, value, qkv_smooth_factor, write_mask, read_mask):
        """
        Smooth QKV
        Args:
            query: [bsz, seq_len, local_heads, qk_dim]
            key: [bsz, seq_len, local_heads, qk_dim]
            value: [bsz, seq_len, local_heads, v_dim]
            qkv_smooth_factor: [local_heads, 3]
            write_mask: [bsz, seq_len, local_heads]
            read_mask: [bsz, seq_len, local_heads]
        Return:
            query: [bsz, seq_len, local_heads, qk_dim]
            key: [bsz, seq_len, local_heads, qk_dim]
            value: [bsz, seq_len, local_heads, v_dim]
        """
        query = ema_smooth_scan(query, qkv_smooth_factor[:,0], read_mask)
        key = ema_smooth_scan(key, qkv_smooth_factor[:,1], write_mask)
        value = ema_smooth_scan(value, qkv_smooth_factor[:,2], write_mask)
        return query, key, value

    
    def forward_state_update_read(self, query, key, value, alpha, beta, write_mask, read_mask):
        """
        Forward pass for the state update read component using the Gated Delta Rule.
        Args:
            query: [bsz, seq_len, local_heads, query_key_dim]
            key: [bsz, seq_len, local_heads, query_key_dim]
            value: [bsz, seq_len, local_heads, value_dim]
            alpha: [bsz, seq_len, local_heads] 
            beta: [bsz, seq_len, local_heads]
            write_mask: [bsz, seq_len, local_heads]
            read_mask: [bsz, seq_len, local_heads]
        Return:
            core_attn_out: [bsz, seq_len, local_heads, v_dim]
        """
        nvtx_range_push(suffix="gated_delta_rule")
        if self.config.deterministic_mode:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=alpha,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=alpha,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        nvtx_range_pop(suffix="gated_delta_rule")
        return core_attn_out, last_recurrent_state

    def forward_project_rms_proj_sum_out(self, core_attn_out, gate, read_score, read_mask):
        """
        Project RMS Proj Sum Out
        Args:
            core_attn_out: [bsz, seq_len, local_heads, value_dim]
            gate: [bsz, seq_len, local_heads, value_dim]
            read_score: [bsz, seq_len, local_heads]
            read_mask: [bsz, seq_len, local_heads]
        Return:
            out: [bsz, seq_len, hidden_size]
        """
        bsz, seq_len, local_heads, value_dim = core_attn_out.shape
        gate = self.act_fn(gate) * read_score.unsqueeze(-1)
        # gate.masked_fill_(read_mask.unsqueeze(-1) == 0, 0.0)
        core_attn_out = self.out_norm(core_attn_out) 
        out = core_attn_out * gate
        # out = out.masked_fill(read_mask.unsqueeze(-1) == 0, 0.0)
        out = out * read_mask.unsqueeze(-1)
        out = out.reshape(bsz, seq_len, -1) # [bsz, seq_len, local_heads*value_dim]
        output, output_bias = self.out_proj(out) # tp AllReduce here
        return output.transpose(0, 1), output_bias  # output: [seq_len, bsz, hidden_size]; bias: [hidden_size] or None

    def forward(self, *args, **kwargs):
        # with torch.autograd.detect_anomaly():
        return self.forward_impl(*args, **kwargs)
    
    def forward_impl(
        self,
        hidden_states,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        # TODO: Deal with attention_mask
        assert inference_context is None, "not support inference_context"

        seq_len, bsz, _ = hidden_states.shape
        hidden_states = hidden_states.transpose(0, 1) # [bsz, seq_len, hidden_dim]
        
        # route, all shapes are [bsz, seq_len, local_heads]
        write_score, write_mask, read_score, read_mask = self.forward_local_write_read_gate(hidden_states)

        # compute input projection
        # q, k, v: [bsz, seq_len, local_heads, *]
        # gate:    [bsz, seq_len, local_heads, value_dim]
        # alpha:   [bsz, seq_len, local_heads]
        # beta:    [bsz, seq_len, local_heads]
        query, key, value, gate, alpha, beta = self.forward_qkv_gate_alpha_beta(hidden_states)

        # compute local factors
        # qkv_smooth_factor: [local_heads, 3]
        alpha_log, beta, qkv_smooth_factor = self.compute_local_factors(alpha, beta)

        # # mask
        # query = query.masked_fill(read_mask.unsqueeze(-1) == 0, 0.0)
        # key = key.masked_fill(write_mask.unsqueeze(-1) == 0, 0.0)
        # value = value.masked_fill(write_mask.unsqueeze(-1) == 0, 0.0)
        # gate = gate.masked_fill(read_mask.unsqueeze(-1) == 0, 0.0)
        # alpha_log = alpha_log.masked_fill(write_mask == 0, 0.0)
        # beta = beta.masked_fill(write_mask == 0, 0.0)

        # smooth qkv
        # query, key, value = self.smooth_qkv(query, key, value, qkv_smooth_factor, write_mask, read_mask)

        # SiLU  for qkv
        query = self.act_fn(query)
        key = self.act_fn(key)
        value = self.act_fn(value)
        

        # L2 norm for qk, but not v
        if self.use_qk_l2norm:
            query = l2norm(query.contiguous())
            key = l2norm(key.contiguous())

        
        # mask
        # query = query.masked_fill(read_mask.unsqueeze(-1) == 0, 0.0)
        # key = key.masked_fill(write_mask.unsqueeze(-1) == 0, 0.0)
        # value = value.masked_fill(write_mask.unsqueeze(-1) == 0, 0.0)
        # gate = gate.masked_fill(read_mask.unsqueeze(-1) == 0, 0.0)
        # alpha_log = alpha_log.masked_fill(write_mask == 0, 0.0)
        # beta = beta.masked_fill(write_mask == 0, 0.0)

        query = query * read_mask.unsqueeze(-1)
        key = key * write_mask.unsqueeze(-1)
        value = value * write_mask.unsqueeze(-1)
        gate = gate * read_mask.unsqueeze(-1)
        alpha_log = alpha_log * write_mask
        beta = beta * write_mask

        # State Update Read
        core_attn_out, last_recurrent_state = self.forward_state_update_read(
            query, key, value, alpha_log, beta, write_mask, read_mask
        )

        output, output_bias = self.forward_project_rms_proj_sum_out(core_attn_out, gate, read_score, read_mask)

        return output, output_bias


    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                "A_log": 0,
                "dt_bias": 0,
            },  # parameters sharded across TP
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata['dp_cp_group'],
        )
        # Submodules
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            module_sharded_sd = sharded_state_dict_default(
                module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
            )
            sharded_state_dict.update(module_sharded_sd)

        # At this point the TP sharding is correctly defined for each tensor, but some of the
        # tensors must be additionally split into separate parts
        in_proj_dim_local_tp = self.in_proj_dim // self.tp_size
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local_tp, (
            in_proj_dim_local_tp,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )

        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim // self.tp_size,
                self.qk_dim // self.tp_size,
                self.v_dim // self.tp_size,
                self.v_dim // self.tp_size,
                self.num_total_heads // self.tp_size,
                self.num_total_heads // self.tp_size,
            ],
            ["query", "key", "value", "z", "beta", "alpha"],
            0,
        )

        return sharded_state_dict

    def backward_dw(self):
        """Execute weight gradient computation for all linear layers."""
        self._backward_in_proj()
        self._backward_out_proj()

    def _backward_in_proj(self):
        """Computes weight gradients of input projection layer."""
        self.in_proj.backward_dw()

    def _backward_out_proj(self):
        """Computes weight gradients of output projection layer."""
        self.out_proj.backward_dw()


    def update_expert_bias(self):
        """Update write/read expert bias for loss-free load balancing.
        Should be called once per optimizer step, after all micro-batches,
        analogous to finalize_model_grads._update_router_expert_bias().
        """
        if not self.enable_expert_bias:
            return
        from megatron.core import parallel_state
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        heads_per_rank = self.num_routed_heads // self.tp_size
        local_start = tp_rank * heads_per_rank
        with torch.no_grad():
            for tokens_per_head, expert_bias in (
                (self.write_local_tokens_per_head, self.write_expert_bias),
                (self.read_local_tokens_per_head,  self.read_expert_bias),
            ):
                # All-reduce across DP+CP only; each TP rank accumulates its own local heads,
                # so including TP would multiply counts by tp_size erroneously.
                torch.distributed.all_reduce(
                    tokens_per_head,
                    group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                )
                avg_tokens = tokens_per_head.sum() / tokens_per_head.shape[0]
                offset = avg_tokens - tokens_per_head  # positive: underloaded
                # Update only the local TP slice of the global expert_bias buffer
                expert_bias[local_start:local_start + heads_per_rank].add_(
                    torch.sign(offset) * self.expert_bias_update_rate
                )
            # Reset accumulators for next step
            self.write_local_tokens_per_head.zero_()
            self.read_local_tokens_per_head.zero_()

def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: List[int], split_names: List[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )

    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim], (
            split_start,
            orig_sh_ten_no_data.local_shape[split_dim],
        )
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel(), (
            chunk_sh_tens,
            t.shape,
        )
        return chunk_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        orig_sh_ten.key, orig_sh_ten.data, sh_ten_build_fn, sh_ten_merge_fn, orig_sh_ten.replica_id
    )


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    # pylint: disable=line-too-long
    '''
    Torch-native implementation of chunked gated delta rule for deterministic mode.
    Need this because FLA is not deterministic.

    Reference: https://github.com/huggingface/transformers/blob/144c8ce2809a2e21914017652700e1ecb450501e/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L470-L547
    Args:
        query: [batch, seq_len, num_heads, qk_dim], num_heads 是专家数量
        key: [batch, seq_len, num_heads, qk_dim]
        value: [batch, seq_len, num_heads, v_dim]
        g: [batch, seq_len, num_heads]
        beta: [batch, seq_len, num_heads]
    对于无需写入的位置, 在序列维度 g 和beta 设为 0
    对于无需读取的位置，在序列维度 q 设为 0
    '''

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state




# ──────────────────────────────────────────────────────────────────────────────
# 关联算子 prefix scan（Hillis-Steele 算法，unchanged）
# ──────────────────────────────────────────────────────────────────────────────

def _assoc_scan(A: Tensor, B: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Hillis-Steele inclusive prefix scan for the linear recurrence:
        s_t = A_t * s_{t-1} + B_t,   s_{-1} = 0

    Associative operator:
        (a1, b1) ⊕ (a2, b2) = (a2 * a1,  a2 * b1 + b2)

    After the scan, position t holds the cumulative composition of 0 .. t:
        s_t = A_cum[t] * s_init + B_cum[t]
    With s_init = 0:  s_t = B_cum[t].

    Complexity:  O(S log S) work,  O(log S) depth.

    A and B may have different trailing dimensions that broadcast:
        e.g., A: (..., 1, S) and B: (..., D, S).

    Args:
        A, B : tensors whose last dimension is the sequence length S
    Returns:
        A_cum, B_cum : same shapes as A and B (computed in float32 for stability)
    """
    S = A.shape[-1]
    a = A.float()   # promote for numerical stability (A gets multiplied log2(S) times)
    b = B.float()

    stride = 1
    while stride < S:
        # Left-pad by `stride` with identity element (1, 0); drop rightmost `stride` elems
        a_left = F.pad(a[..., :-stride], (stride, 0), value=1.0)   # (..., S)
        b_left = F.pad(b[..., :-stride], (stride, 0), value=0.0)   # (..., S)

        # Compose  (a_left, b_left) ⊕ (a, b):
        #   new_a = a * a_left
        #   new_b = a * b_left + b          ← must use OLD a; update b first
        b = a * b_left + b
        a = a * a_left

        stride <<= 1

    return a, b

# ──────────────────────────────────────────────────────────────────────────────
# 辅助：沿最后维度的 inclusive prefix max（Hillis-Steele）
# ──────────────────────────────────────────────────────────────────────────────

def _prefix_max_seq(x: Tensor) -> Tensor:
    """
    Inclusive parallel prefix max over the last dimension.
    Identity element = -inf (positions before the first active are -inf).

    Complexity: O(S log S) work, O(log S) depth.
    """
    S = x.shape[-1]
    result = x.clone().float()
    stride = 1
    while stride < S:
        left = F.pad(result[..., :-stride], (stride, 0), value=float('-inf'))
        result = torch.maximum(result, left)
        stride <<= 1
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 实现 2: 并行关联扫描（O(log S) 深度）
# ──────────────────────────────────────────────────────────────────────────────


def ema_smooth_scan(
    x: Tensor,      # (B, S, H, D)
    lam: Tensor,    # (H)
    mask: Tensor,   # (B, S, H) bool
) -> Tensor:        # (B, S, H, D)
    """
    Parallel EMA smoothing with decaying mask via associative prefix scan.
    O(S log S) work, O(log S) depth.  See module docstring for algorithm.
    """
    B, S, H, D = x.shape
    mask_bhs = mask.permute(0, 2, 1)                          # (B, H, S)

    # ── Step 1: last active index STRICTLY before t ─────────────────────────
    arange_S  = torch.arange(S, device=x.device, dtype=torch.float32)

    # masked_idx: t where active, -inf where inactive
    masked_idx = torch.where(
        mask_bhs,
        arange_S[None, None, :].expand(B, H, S),
        torch.full((1, 1, 1), float('-inf'), device=x.device).expand(B, H, S),
    )                                                         # (B, H, S)

    L_inc  = _prefix_max_seq(masked_idx)                      # inclusive, (B, H, S)

    # Exclusive prefix max: shift right by 1; clamp -inf → -1 (virtual start)
    L_excl = torch.cat([
        torch.full((B, H, 1), -1.0, device=x.device),
        L_inc[..., :-1],
    ], dim=-1)                                                # (B, H, S)
    L_excl = torch.clamp(L_excl, min=-1.0)                   # replace any -inf with -1

    # ── Step 2: delta_T[t] = t - L_excl[t]  (>= 1 always) ─────────────────
    delta_T = arange_S[None, None, :].expand(B, H, S) - L_excl   # (B, H, S)
    alpha   = lam.float()[None, :, None] ** delta_T.float()          # (B, H, S)

    # ── Step 3: linear-recurrence coefficients ──────────────────────────────
    #   active  (mask=1): A = alpha,  B = (1-alpha)*x_t  → new committed state
    #   inactive(mask=0): A = 1,      B = 0              → propagate last commit
    x_bHDS     = x.permute(0, 2, 3, 1).contiguous()               # (B, H, D, S)
    alpha_bH1S = alpha[:, :, None, :]                              # (B, H, 1, S)
    m_bH1S     = mask_bhs[:, :, None, :].float()                   # (B, H, 1, S)

    A_scan = torch.where(
        mask_bhs[:, :, None, :],
        alpha_bH1S,
        torch.ones_like(alpha_bH1S),
    )                                                              # (B, H, 1, S)
    B_scan = m_bH1S * (1.0 - alpha_bH1S) * x_bHDS                 # (B, H, D, S)

    # ── Step 4: associative scan ─────────────────────────────────────────────
    _, B_cum = _assoc_scan(A_scan, B_scan)                         # (B, H, D, S)
    s_scan   = B_cum.to(x.dtype).permute(0, 3, 1, 2)              # (B, S, H, D)

    # ── Step 5: final output ─────────────────────────────────────────────────
    #   active:  y_t = s_scan[t]                    (scan result IS the committed output)
    #   inactive:y_t = alpha_t * s_scan[t] + (1-alpha_t) * x_t
    #                  (s_scan[t] for inactive = propagated last-commit state)
    alpha_bSH1 = alpha.permute(0, 2, 1).unsqueeze(-1).to(x.dtype) # (B, S, H, 1)
    mask_bSH1  = mask.unsqueeze(-1)                                # (B, S, H, 1)

    y = torch.where(
        mask_bSH1,
        s_scan,
        alpha_bSH1 * s_scan + (1.0 - alpha_bSH1) * x,
    )
    return y                                                       # (B, S, H, D)



class _EMASmoothFunc(torch.autograd.Function):
    """Custom autograd function that uses recomputation for backward.

    Forward: runs ema_smooth_scan under torch.no_grad(), saves only inputs.
    Backward: re-runs ema_smooth_scan with grad enabled, then calls
              torch.autograd.grad to obtain input gradients.
    This avoids retaining the O(S log S) intermediate computation graph.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,      # (B, S, H, D)
        lam: Tensor,    # (B, H)
        mask: Tensor,   # (B, S, H) bool
    ) -> Tensor:
        ctx.save_for_backward(x, lam, mask)
        with torch.no_grad():
            y = ema_smooth_scan(x, lam, mask)
        return y

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, lam, mask = ctx.saved_tensors

        # Recompute forward with autograd graph
        with torch.enable_grad():
            x_recomp = x.detach().requires_grad_(True)
            lam_recomp = lam.detach().requires_grad_(True)
            y_recomp = ema_smooth_scan(x_recomp, lam_recomp, mask)

        grad_x, grad_lam = torch.autograd.grad(
            y_recomp, (x_recomp, lam_recomp), grad_output,
        )
        return grad_x, grad_lam, None  # None for mask (not differentiable)


def ema_smooth(
    x: Tensor,      # (B, S, H, D)
    lam: Tensor,    # (B, H)
    mask: Tensor,   # (B, S, H) bool
) -> Tensor:        # (B, S, H, D)
    """EMA smoothing operator with memory-efficient recomputation backward.

    Numerically equivalent to ema_smooth_scan in forward.
    Backward uses recomputation: only inputs are saved, intermediate
    activations from the associative scan are discarded and recomputed
    during the backward pass, reducing peak memory from O(B*S*H*D*log S)
    to O(B*S*H*D).

    Args:
        x:    (B, S, H, D) feature tensor to smooth.
        lam:  (B, H) per-(batch, head) EMA decay in [0, 1].
        mask: (B, S, H) bool; True = commit / write gate open.
    Returns:
        y:    (B, S, H, D) smoothed output.
    """
    return _EMASmoothFunc.apply(x, lam, mask)