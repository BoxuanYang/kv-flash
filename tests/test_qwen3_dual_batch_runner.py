"""CPU-only orchestration checks; these do not validate CUDA kernels or KV glue.

Run: python -m unittest discover -s tests -p test_qwen3_dual_batch_runner.py -v
"""

import importlib.util
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

# Stub unavailable native dependencies only in this test module. Production
# code imports torch, KT and SGLang normally at the top of its file.
spec = importlib.util.spec_from_file_location(
    "_dual_batch_test_module", Path(__file__).resolve().parents[1] / "Qwen3DualBatchRunner.py"
)
runner_module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {
    spec.name: runner_module,
    "torch": ModuleType("torch"),
    "kt_kernel.experts_base": SimpleNamespace(BaseMoEWrapper=object),
    "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(ForwardBatch=object),
}):
    spec.loader.exec_module(runner_module)
Qwen3DualBatchRunner = runner_module.Qwen3DualBatchRunner


class ConcurrentProbe(Qwen3DualBatchRunner):
    """Execute synthetic dependent stages on two workers, using the real loop."""

    def __init__(self, layers):
        self.layer_num = layers
        self.trace = []
        self.pending = {}
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.completed_layer = {(kind, mb): -1 for kind in ("attn", "moe") for mb in (0, 1)}
        self.values = [2, 7]
        self.overlap_pairs = 0
        self.stream = SimpleNamespace(synchronize=self._barrier)
        self.moe_collected = False

    def _submit(self, kind, mb, layer):
        other = "moe" if kind == "attn" else "attn"
        predecessor = layer - 1 if kind == "attn" else layer
        if self.completed_layer[other, mb] != predecessor:
            raise AssertionError("A stage started before its data dependency completed")
        if self.completed_layer[kind, mb] != layer - 1:
            raise AssertionError("A layer was skipped or repeated")
        if kind in self.pending:
            raise AssertionError("A stage was reused before synchronization")
        self.trace.append(("submit", kind, mb, layer))
        started, release = Event(), Event()

        def work():
            started.set()
            if not release.wait(5):
                raise AssertionError("Stage was never released")
            x = self.values[mb]
            self.values[mb] = (3 * x + layer) if kind == "attn" else (2 * x - layer)

        self.pending[kind] = (mb, layer, started, release, self.pool.submit(work))

    def _sync(self, kind, mb):
        entry = self.pending[kind]
        if entry[0] != mb:
            raise AssertionError("Synchronized the wrong microbatch")
        # Startup/drain have one task. Every other first sync must see both
        # workers started and held in-flight together; no timing/sleep heuristic.
        _, layer, _, _, future = entry
        first_sync = not entry[3].is_set()
        if first_sync:
            boundary = (kind, mb, layer) in (
                ("attn", 0, 0), ("moe", 1, self.layer_num - 1)
            )
            if len(self.pending) != (1 if boundary else 2):
                raise AssertionError("Wait happened before both paired submissions")
            for _, _, started, _, _ in self.pending.values():
                if not started.wait(5):
                    raise AssertionError("The two stages did not start concurrently")
            if not boundary:
                self.overlap_pairs += 1
            for _, _, _, release, _ in self.pending.values():
                release.set()
        future.result(timeout=5)
        self.completed_layer[kind, mb] = layer
        self.trace.append(("sync", kind, mb, layer))
        del self.pending[kind]

    def submit_attn(self, mb, layer):
        self._submit("attn", mb, layer)

    def submit_moe(self, mb, layer):
        self._submit("moe", mb, layer)

    def sync_moe(self, mb, layer):
        assert self.pending["moe"][:2] == (mb, layer)
        self.moe_collected = True

    def _barrier(self):
        if "moe" in self.pending:
            assert self.moe_collected, "MoE result was not collected before the barrier"
        for kind in list(self.pending):
            self._sync(kind, self.pending[kind][0])
        self.moe_collected = False

    def close(self):
        for _, _, _, release, _ in self.pending.values():
            release.set()
        self.pool.shutdown(wait=True)


class Scalar:
    """Mutable stand-in for a tensor, exposing residual/staging alias bugs."""

    def __init__(self, value):
        self.value = value

    def clone(self):
        return Scalar(self.value)


def norm(x, residual=None):
    if residual is None:
        return Scalar(x.value / 2)
    residual.value += x.value  # Like fused add+norm, deliberately mutate residual.
    return Scalar(residual.value / 2), residual


class SharedBufferMoE:
    def __init__(self, layer, shared):
        self.layer = layer
        self.shared = shared

    def submit_forward(self, x, ids, weights, stream):
        assert self.shared.pending is None
        self.shared.calls.append(("submit", stream))
        self.shared.pending = (self.layer, x, ids, weights)

    def sync_forward(self, x, stream):
        layer, submitted_x, ids, weights = self.shared.pending
        assert layer == self.layer and submitted_x is x
        self.shared.calls.append(("sync", stream))
        self.shared.output.value = x.value * weights.value + ids.value + layer
        self.shared.pending = None
        return self.shared.output


def scalar_runner(layer_num):
    """Use real runner stages with fake compute/streams; no torch dependency."""
    runner = object.__new__(Qwen3DualBatchRunner)
    runner.layer_num = layer_num
    runner_module.torch = SimpleNamespace(cuda=SimpleNamespace())
    runner.stream = SimpleNamespace(cuda_stream=123, synchronize=lambda: None)
    runner.layers = []
    runner.attn_batches = []
    for i in range(layer_num):
        def attention(*, positions, hidden_states, forward_batch, offset=i):
            runner.attn_batches.append(forward_batch)
            return Scalar(hidden_states.value * 3 + positions + offset)

        runner.layers.append(SimpleNamespace(
            input_layernorm=norm,
            self_attn=attention,
            post_attention_layernorm=norm,
            mlp=SimpleNamespace(
                gate=lambda x: (x, None),
                topk=lambda x, logits: SimpleNamespace(
                    topk_ids=Scalar(1), topk_weights=Scalar(2)
                ),
            ),
        ))
    # Deliberately share one output slot across all layers and microbatches.
    shared = SimpleNamespace(output=Scalar(0), pending=None, calls=[])
    runner.cpu_moe = [SharedBufferMoE(i, shared) for i in range(layer_num)]
    return runner, shared.output


class PipelineTests(unittest.TestCase):
    def test_forward_embeds_batch_input_ids_and_preserves_metadata(self):
        # Exercise the public ForwardBatch-shaped interface without importing
        # SGLang/CUDA. This is not an end-to-end SGLang integration test.
        runner, _ = scalar_runner(2)
        runner.device = "cuda:0"
        cuda = runner_module.torch.cuda
        caller = SimpleNamespace(cuda_stream=321, synchronize=Mock())
        cuda.current_stream = lambda _: caller
        runner_module.torch.bfloat16 = "bf16"

        embedded = []

        class Embedding:
            weight = SimpleNamespace(dtype="bf16")

            def __call__(self, ids):
                embedded.append(ids)
                return Scalar(ids.value * 10)

        runner.model = SimpleNamespace(embed_tokens=Embedding(), norm=norm)
        batches = [
            SimpleNamespace(
                input_ids=SimpleNamespace(value=x, device="cuda:0", shape=(1,)),
                positions=SimpleNamespace(device="cuda:0", shape=(1,)),
                batch_size=1,
                forward_mode=SimpleNamespace(is_decode=lambda: True),
                attn_backend=object(),
            )
            for x in (2, 7)
        ]
        # The scalar attention uses a numeric position, retaining the tensor
        # attributes required by the actual forward input validation.
        class Position(int):
            device = "cuda:0"
            shape = (1,)

        for batch in batches:
            batch.positions = Position(1024)
        outputs = runner.forward(*batches)
        self.assertIs(runner.stream, caller)
        self.assertEqual(caller.synchronize.call_count, 5)  # 两层对应五个时间槽。
        self.assertTrue(all(stream == 321 for _, stream in runner.cpu_moe[0].shared.calls))
        self.assertEqual(embedded, [b.input_ids for b in batches])
        self.assertEqual(runner.attn_batches, batches * 2)
        for output, token in zip(outputs, (2, 7)):
            hidden, residual = token * 10, None
            for layer in range(2):
                residual = hidden if residual is None else residual + hidden
                residual += 3 * (residual / 2) + 1024 + layer
                hidden = residual + 1 + layer
            self.assertEqual(output.value, (hidden + residual) / 2)

        # 连续 decode 时也必须获取调用方本次的流，不能沿用上一次的流。
        previous_outputs = [x.value for x in outputs]
        next_caller = SimpleNamespace(cuda_stream=456, synchronize=Mock())
        cuda.current_stream = lambda _: next_caller
        runner.cpu_moe[0].shared.calls.clear()
        next_outputs = runner.forward(*batches)
        self.assertIs(runner.stream, next_caller)
        self.assertEqual(next_caller.synchronize.call_count, 5)
        self.assertTrue(all(stream == 456 for _, stream in runner.cpu_moe[0].shared.calls))
        self.assertEqual([x.value for x in outputs], previous_outputs)
        self.assertEqual([x.value for x in next_outputs], previous_outputs)

    def test_concurrent_pairs_and_serial_equivalence(self):
        for layer_num in (1, 2, 3, 48):
            with self.subTest(layers=layer_num):
                probe = ConcurrentProbe(layer_num)
                try:
                    probe._run_pipeline()
                    expected = [2, 7]
                    for i in range(layer_num):
                        expected = [2 * (3 * x + i) - i for x in expected]
                    self.assertEqual(probe.values, expected)
                    self.assertEqual(probe.overlap_pairs, 2 * layer_num - 1)
                    self.assertFalse(probe.pending)
                    self.assertTrue(all(v == layer_num - 1 for v in probe.completed_layer.values()))
                    self.assertEqual(len(probe.trace), 8 * layer_num)
                finally:
                    probe.close()

    def test_two_layer_submit_order(self):
        probe = ConcurrentProbe(2)
        try:
            probe._run_pipeline()
            self.assertEqual([row[1:] for row in probe.trace if row[0] == "submit"], [
                ("attn", 0, 0),
                ("moe", 0, 0), ("attn", 1, 0),
                ("moe", 1, 0), ("attn", 0, 1),
                ("moe", 0, 1), ("attn", 1, 1),
                ("moe", 1, 1),
            ])
        finally:
            probe.close()

    def test_real_stage_residual_flow_and_shared_output_ownership(self):
        runner, shared = scalar_runner(3)
        for _ in range(2):  # Fresh state for successive forwards.
            inputs = [Scalar(x) for x in (2, 7)]
            runner.batches = [SimpleNamespace(positions=1024) for _ in (0, 1)]
            runner.hidden_states = inputs.copy()
            runner.residual = [None, None]
            runner.topk_ids = [None, None]
            runner.topk_weights = [None, None]
            runner._run_pipeline()
            shared.value = -999  # Neither activation may alias KT's shared output.
            for mb, initial in enumerate((2, 7)):
                residual, hidden = None, initial
                for layer in range(3):
                    residual = hidden if residual is None else residual + hidden
                    attn = 3 * (residual / 2) + 1024 + layer
                    residual += attn
                    hidden = 2 * (residual / 2) + 1 + layer
                self.assertEqual(runner.hidden_states[mb].value, hidden)
                self.assertEqual(runner.residual[mb].value, residual)
                self.assertIsNone(runner.topk_ids[mb])
                self.assertIsNone(runner.topk_weights[mb])
                self.assertIsNot(runner.residual[0], runner.residual[1])
        self.assertEqual(runner.cpu_moe[0].shared.calls, [(op, 123) for _ in range(12)
                                                       for op in ("submit", "sync")])


if __name__ == "__main__":
    unittest.main()
