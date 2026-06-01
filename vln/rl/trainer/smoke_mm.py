"""3b-3/3b-4: full GRPO step on a real multimodal parquet batch.
infer_batch (old_log_probs) -> inject advantages -> train_batch (ppo_loss),
with pixel_values + image_grid_thw + 3-D mrope position_ids.

  docker exec verl-dev bash -lc "cd /workspace/WorldModel && PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0,1,2,3 VLN_FSDP_SIZE=4 \
    VLN_CKPT=/workspace/WorldModel/checkpoints/Qwen3-VL-4B-vln-r2r-merged \
    VLN_PARQUET=/workspace/rl_test/frames_test.parquet \
    VLN_FRAMES=/workspace/rl_test/frames \
    python -m vln.rl.trainer.smoke_mm"
"""
import os
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

from vln.rl.trainer.advantage import compute_grpo_advantages
from vln.rl.trainer.collate import make_example, to_verl_batch
from vln.rl.trainer.smoke_engine import build_config


def main():
    ckpt = os.environ["VLN_CKPT"]
    fsdp_size = int(os.environ.get("VLN_FSDP_SIZE", "4"))
    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id

    df = pd.read_parquet(os.environ["VLN_PARQUET"])
    df, _ = compute_grpo_advantages(df)
    idxs = list(range(fsdp_size))  # B == fsdp_size (divisible)
    examples = [make_example(df.iloc[i], os.environ["VLN_FRAMES"], proc, float(df.iloc[i]["advantage"])) for i in idxs]
    batch = to_verl_batch(examples, pad_id, proc)

    ray.init()
    wg = RayWorkerGroup(
        resource_pool=RayResourcePool(process_on_nodes=[fsdp_size]),
        ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=build_config(ckpt, fsdp_size)),
    )
    wg.reset()

    global_token_num = batch["attention_mask"].sum(-1).tolist()
    data = DataProto.from_single_dict(
        {
            "input_ids": batch["input_ids"],
            "prompts": batch["prompts"],
            "attention_mask": batch["attention_mask"],
            "position_ids": batch["position_ids"],
            "responses": batch["responses"],
            "response_mask": batch["response_mask"],
            "multi_modal_inputs": batch["multi_modal_inputs"],
        },
        meta_info={"temperature": 1.0, "global_token_num": global_token_num, "compute_loss": False},
    )

    # 3b-3: VL forward -> old_log_probs
    data_td = left_right_2_no_padding(data.to_tensordict())
    out = wg.infer_batch(data_td).get()
    old_log_probs = no_padding_2_padding(tu.get(out, "log_probs").cpu(), data_td)
    print("VL forward OK, log_probs shape:", tuple(old_log_probs.shape),
          "responses shape:", tuple(batch["responses"].shape))

    # 3b-4: inject old_log_probs + advantages -> one GRPO train step
    data = data.union(DataProto.from_single_dict({"old_log_probs": old_log_probs}))
    data.batch["advantages"] = batch["advantages"]
    wg.set_loss_fn(partial(ppo_loss, config=ActorConfig(
        strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=-1, use_kl_loss=False)))

    data_td = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(data_td, global_batch_size=data_td.shape[0])
    metrics = tu.get(wg.train_batch(data_td).get(), "metrics")
    peak = metrics.get("perf/max_memory_allocated_gb")
    print("train_batch OK: loss=%s grad_norm=%s peak_mem_gb=%s" % (
        metrics.get("loss"), metrics.get("grad_norm"), peak))
    assert peak is None or peak < 40, f"3b-5 memory smoke: peak {peak}GB exceeds 40G"
    print("3b-4/3b-5 multimodal GRPO step + memory smoke PASSED")


if __name__ == "__main__":
    main()
