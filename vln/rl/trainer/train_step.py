"""7.4-1: one real GRPO train step from a rollout parquet, then save a
vLLM-loadable HF ckpt. Generalizes smoke_mm (which used only B=fsdp_size rows)
to N rows with verl's internal micro-batching.

  docker exec verl-dev bash -lc "cd /workspace/WorldModel && PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m vln.rl.trainer.train_step \
      --ckpt /workspace/WorldModel/checkpoints/Qwen3-VL-4B-vln-r2r-merged \
      --parquet /workspace/rl_test/frames_test.parquet \
      --frames /workspace/rl_test/frames \
      --save-dir /workspace/rl_test/ckpt_step1 \
      --fsdp-size 4 --max-rows 16"
"""
import argparse
from functools import partial

import pandas as pd
import ray
from transformers import AutoProcessor

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from vln.rl.trainer.advantage import assert_single_policy_version, compute_grpo_advantages
from vln.rl.trainer.collate import make_example, to_verl_batch
from vln.rl.trainer.smoke_engine import build_config


def train_step(ckpt, parquet, frames, save_dir, fsdp_size, max_rows, rl_step):
    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id

    df = pd.read_parquet(parquet)
    assert_single_policy_version(df)          # fail-fast on mixed policy_version
    df, stats = compute_grpo_advantages(df)   # GRPO advantage over full parquet groups
    print("group stats:", stats)

    n = min(max_rows, len(df))
    n -= n % fsdp_size                        # dp ranks split the batch
    rows = [df.iloc[i] for i in range(n)]
    examples = [make_example(r, frames, proc, float(r["advantage"])) for r in rows]
    batch = to_verl_batch(examples, pad_id, proc)
    print(f"training on {n} rows (rl_step={rl_step})")

    ray.init()
    wg = RayWorkerGroup(
        resource_pool=RayResourcePool(process_on_nodes=[fsdp_size]),
        ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=build_config(ckpt, fsdp_size)),
    )
    wg.reset()

    data = DataProto.from_single_dict(
        {
            "input_ids": batch["input_ids"], "prompts": batch["prompts"],
            "attention_mask": batch["attention_mask"], "position_ids": batch["position_ids"],
            "responses": batch["responses"], "response_mask": batch["response_mask"],
            "multi_modal_inputs": batch["multi_modal_inputs"],
        },
        meta_info={"temperature": 1.0,
                   "global_token_num": batch["attention_mask"].sum(-1).tolist(),
                   "compute_loss": False},
    )

    data_td = left_right_2_no_padding(data.to_tensordict())
    old_lp = no_padding_2_padding(tu.get(wg.infer_batch(data_td).get(), "log_probs").cpu(), data_td)
    data = data.union(DataProto.from_single_dict({"old_log_probs": old_lp}))
    data.batch["advantages"] = batch["advantages"]

    wg.set_loss_fn(partial(ppo_loss, config=ActorConfig(
        strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=-1, use_kl_loss=False)))
    data_td = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(data_td, global_batch_size=data_td.shape[0])
    metrics = tu.get(wg.train_batch(data_td).get(), "metrics")
    print("train_batch: loss=%s grad_norm=%s peak_gb=%s" % (
        metrics.get("loss"), metrics.get("grad_norm"), metrics.get("perf/max_memory_allocated_gb")))

    wg.save_checkpoint(save_dir, global_step=rl_step + 1, max_ckpt_to_keep=3)
    print(f"saved ckpt -> {save_dir}/huggingface (policy_version=step{rl_step + 1})")
    print("7.4-1 train_step PASSED")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--fsdp-size", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=16)
    p.add_argument("--rl-step", type=int, default=0)
    a = p.parse_args()
    train_step(a.ckpt, a.parquet, a.frames, a.save_dir, a.fsdp_size, a.max_rows, a.rl_step)


if __name__ == "__main__":
    main()
