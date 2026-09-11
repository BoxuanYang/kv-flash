# SPDX-License-Identifier: Apache-2.0
"""单 GPU、Eager 模式下，使用两个固定 microbatch 执行 Qwen3 MoE decode。

复用已加载的 SGLang Qwen3MoeModel 主干中的 Attention、Norm 和 router，
不调用 decoder.forward()、分布式通信模块或同步 MoE forward。
每层传入一个已加载的 KT 推理 wrapper，所有路由专家均在 CPU 上执行，
通过 submit_forward()/sync_forward() 提交计算并获取结果。

在同一 CUDA stream 中，先排入 MoE 输入传输和提交 callback，再排入配对的
GPU Attention，最后等待 CPU MoE 完成并将结果传回 GPU。

调用方提供两个 decode 模式的 SGLang ForwardBatch，各自具有独立且已初始化的
Attention backend/metadata，以及有效的 GPU KV 映射。本模块执行 embedding，
并按输入顺序返回两份经过最终归一化的 hidden states。
Batch 切分、KV 分配、LM head、采样和 scheduler 接入由外部负责。

forward() 期间独占 KT CPUInfer 队列及其类级共享暂存 buffer。
同一时间最多有一个 CPU MoE 任务在执行；每组并行任务完成后才进入下一组，
复用暂存槽位前会复制 KT 输出。本模块不支持 CUDA Graph capture，
也不额外持有一份模型权重。
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from kt_kernel.experts_base import BaseMoEWrapper
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class Qwen3DualBatchRunner:
    """先执行 A(0, i) || M(1, i-1)，再执行 A(1, i) || M(0, i)。

    参数：
        model: 已加载的 Qwen3MoeModel 主干，不包含 Qwen3MoeForCausalLM 外层。
        cpu_moe: 按层排列、已加载且兼容 BaseMoEWrapper 的 KT wrapper。
            传入底层 KT wrapper，不经过 KTEPWrapper.apply()。
        device: 存放 Attention 权重及两个 microbatch 的 GPU。
    """

    def __init__(
        self,
        model: Any,
        cpu_moe: Sequence[BaseMoEWrapper],
        device: str = "cuda:0",
    ):
        self.model = model
        self.layers = model.layers
        self.cpu_moe = tuple(cpu_moe)
        self.layer_num = len(self.layers)
        if not self.layer_num or len(self.cpu_moe) != self.layer_num:
            raise ValueError("Supply one loaded KT wrapper per decoder layer")
        self._validate_model()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Qwen3DualBatchRunner requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

    def _validate_model(self):
        if getattr(self.model, "start_layer", 0) != 0 or getattr(
            self.model, "end_layer", self.layer_num
        ) != self.layer_num:
            raise ValueError("Pipeline-parallel model partitions are unsupported")
        pp_group = getattr(self.model, "pp_group", None)
        if pp_group is not None and pp_group.world_size != 1:
            raise ValueError("Only single-rank models are supported")
        for i, (layer, wrapper) in enumerate(zip(self.layers, self.cpu_moe)):
            if layer.layer_id != i or wrapper.layer_idx != i:
                raise ValueError("Model layers and KT wrappers must use matching layer IDs")
            if (
                layer.attn_tp_size != 1
                or layer.mlp.tp_size != 1
                or getattr(layer.mlp, "ep_size", 1) != 1
            ):
                raise ValueError("TP and EP must both be 1")
            if wrapper.num_gpu_experts != 0:
                raise ValueError("This runner requires all routed experts on CPU")
            if wrapper.max_deferred_experts_per_token != 0:
                raise ValueError("Deferred experts are unsupported by this exact decode path")
            if wrapper.cpu_infer is not self.cpu_moe[0].cpu_infer or wrapper.moe is None:
                raise ValueError("KT experts must be loaded and share this runner's CPUInfer")

    def _validate_inputs(self, batches):
        if batches[0].attn_backend is batches[1].attn_backend:
            raise ValueError("Each microbatch needs its own initialized attention backend")
        if self.model.embed_tokens.weight.dtype != torch.bfloat16:
            raise ValueError("The current KT staging path requires BF16 hidden states")
        for batch in batches:
            if batch.input_ids.device != self.device or batch.positions.device != self.device:
                raise ValueError("Both microbatches must reside on the runner's GPU")
            if (
                batch.batch_size <= 0
                or batch.input_ids.shape != (batch.batch_size,)
                or batch.positions.shape != (batch.batch_size,)
            ):
                raise ValueError("Decode requires one position and one token per request")
            if not batch.forward_mode.is_decode():
                raise ValueError("Only ordinary decode is supported")
        if batches[0].batch_size != batches[1].batch_size:
            raise ValueError("The first version requires two equally sized microbatches")

    def forward(self, m0: ForwardBatch, m1: ForwardBatch):
        """执行一个 decode step；调用方负责启用推理模式并准备 KV metadata。

        沿用调用方当前 CUDA stream，输入和权重须已在该流上准备。
        每个流水线时间槽结束时同步；最终 Norm 的结果仍由该流按序消费。
        同一个 runner 不支持多个 forward 同时执行。
        """
        self._validate_inputs((m0, m1))
        self.stream = torch.cuda.current_stream(self.device)
        self.batches = (m0, m1)
        self.hidden_states = [self.model.embed_tokens(b.input_ids) for b in self.batches]
        self.residual = [None, None]
        self.topk_ids = [None, None]
        self.topk_weights = [None, None]

        self._run_pipeline()
        return tuple(
            self.model.norm(self.hidden_states[mb], self.residual[mb])[0]
            for mb in (0, 1)
        )

    def _run_pipeline(self):
        """每轮对应一个时间槽，依次完成启动、并行执行和收尾。"""
        overlap_steps = 2 * self.layer_num - 1
        for i in range(overlap_steps + 2):
            if i == 0:
                # 启动：GPU 执行 A0(第 0 层)，CPU 空闲。
                self.submit_attn(0, 0)
            elif i == overlap_steps + 1:
                # 收尾：GPU 空闲，CPU 执行 M1(最后一层)。
                self.submit_moe(1, self.layer_num - 1)
                self.sync_moe(1, self.layer_num - 1)
            else:
                # GPU 依次执行：A1(第 0 层)、A0(第 1 层)、A1(第 1 层)、……
                # CPU 依次执行：M0(第 0 层)、M1(第 0 层)、M0(第 1 层)、……
                attn_batch = i % 2
                moe_batch = 1 - attn_batch
                attn_layer = i // 2
                moe_layer = (i - 1) // 2

                self.submit_moe(moe_batch, moe_layer)
                self.submit_attn(attn_batch, attn_layer)
                self.sync_moe(moe_batch, moe_layer)

            # 统一等待本时间槽全部完成，包括 MoE 结果从 CPU 传回 GPU。
            self.stream.synchronize()

    def submit_attn(self, mb: int, layer_idx: int):
        layer = self.layers[layer_idx]
        if self.residual[mb] is None:
            # embedding 输出由本 runner 创建，可直接作为 residual 使用。
            self.residual[mb] = self.hidden_states[mb]
            x = layer.input_layernorm(self.hidden_states[mb])
        else:
            x, self.residual[mb] = layer.input_layernorm(
                self.hidden_states[mb], self.residual[mb]
            )
        x = layer.self_attn(
            positions=self.batches[mb].positions,
            hidden_states=x,
            forward_batch=self.batches[mb],
        )
        x, self.residual[mb] = layer.post_attention_layernorm(x, self.residual[mb])
        router_logits, _ = layer.mlp.gate(x)
        topk = layer.mlp.topk(x, router_logits)
        self.topk_weights[mb], self.topk_ids[mb] = topk.topk_weights, topk.topk_ids
        self.hidden_states[mb] = x

    def submit_moe(self, mb: int, layer_idx: int):
        self.cpu_moe[layer_idx].submit_forward(
            self.hidden_states[mb],
            self.topk_ids[mb],
            self.topk_weights[mb],
            self.stream.cuda_stream,
        )

    def sync_moe(self, mb: int, layer_idx: int):
        """排入 KT 等待和结果复制操作，由循环末尾的同步确保完成。"""
        output = self.cpu_moe[layer_idx].sync_forward(
            self.hidden_states[mb], self.stream.cuda_stream
        )
        # KT 返回共享 buffer，复制结果后才能安全复用该槽位。
        self.hidden_states[mb] = output.clone()
        self.topk_ids[mb] = self.topk_weights[mb] = None
