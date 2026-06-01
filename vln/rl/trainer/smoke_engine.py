"""3b-1 smoke: bring up the verl Ray TrainingWorker + FSDP engine with our
Qwen3-VL-4B ckpt and random TEXT data, run infer_batch (old_log_probs) +
train_batch (ppo_loss). Goal: prove the Ray/FSDP/config link works before
touching prompt/response convention (3b-2) or multimodal (3b-3).

Run in verl-dev docker, single GPU:
  docker exec verl-dev bash -lc "cd /workspace/WorldModel && PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0 VLN_CKPT=/workspace/WorldModel/checkpoints/Qwen3-VL-4B-vln-r2r-merged \
    python -m vln.rl.trainer.smoke_engine"
"""
import os
from functools import partial

import numpy as np
import ray
import torch

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.workers.config import ActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding


def build_config(ckpt, fsdp_size):
    # fsdp_size = #GPUs to shard the actor over. env1 (A100-40G) needs >=2 for
    # full-FT 4B (params+grads+AdamW ~64GB fp32 > 40GB); env2 (GB200/300 big
    # HBM) can run fsdp_size=1. ulysses SP stays 1 (4B needs no seq parallel).
    model_config = HFModelConfig(path=ckpt, use_remove_padding=True)
    engine_config = FSDPEngineConfig(
        forward_only=False, fsdp_size=fsdp_size, strategy="fsdp2", ulysses_sequence_parallel_size=1,
        param_offload=True, optimizer_offload=True, grad_offload=True,
        use_dynamic_bsz=True, use_remove_padding=True,
        max_token_len_per_gpu=8192, infer_max_token_len_per_gpu=8192,  # >= max VL seq len (9-frame prompt ~2.3k+)
    )
    return TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=FSDPOptimizerConfig(),
        # hf_model -> save_checkpoint writes a vLLM-loadable HF dir under <path>/huggingface/
        checkpoint_config=CheckpointConfig(save_contents=["model", "hf_model"]),
    )


def main():
    ckpt = os.environ["VLN_CKPT"]
    fsdp_size = int(os.environ.get("VLN_FSDP_SIZE", "1"))
    ray.init()
    config = build_config(ckpt, fsdp_size)
    wg = RayWorkerGroup(
        resource_pool=RayResourcePool(process_on_nodes=[fsdp_size]),
        ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=config),
    )
    wg.reset()

    torch.manual_seed(0)
    np.random.seed(0)
    B, L = 8, 16  # B must be divisible by fsdp_size (dp ranks split the batch)
    rlen = L // 2
    vocab = config.model_config.hf_config.get_text_config().vocab_size  # VL config nests vocab in text_config
    input_ids = torch.randint(0, vocab, (B, L))
    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_valid_token=0.8, max_ratio_of_left_padding=0.2,
        min_ratio_of_valid_token=0.6,
    )
    position_ids = compute_position_id_with_mask(attention_mask)
    global_token_num = torch.sum(attention_mask, dim=-1).tolist()
    responses = input_ids[:, rlen:]
    response_mask = attention_mask[:, rlen:]

    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :rlen],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
        },
        meta_info={"temperature": 1.0, "global_token_num": global_token_num, "compute_loss": False},
    )

    # old_log_probs via pre-forward
    data_td = left_right_2_no_padding(data.to_tensordict())
    out = wg.infer_batch(data_td).get()
    logprobs_unpad = tu.get(out, "log_probs").cpu()
    from verl.workers.utils.padding import no_padding_2_padding
    old_log_probs = no_padding_2_padding(logprobs_unpad, data_td)
    print("infer_batch OK, old_log_probs shape:", tuple(old_log_probs.shape))

    data = data.union(DataProto.from_single_dict({"old_log_probs": old_log_probs}))
    data.batch["advantages"] = torch.rand_like(responses, dtype=torch.float32)

    actor_config = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=-1, use_kl_loss=False)
    wg.set_loss_fn(partial(ppo_loss, config=actor_config))

    data_td = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(data_td, global_batch_size=data_td.shape[0])
    metrics = tu.get(wg.train_batch(data_td).get(), "metrics")
    print("train_batch OK, metrics:", metrics)
    print("3b-1 smoke PASSED")


if __name__ == "__main__":
    main()
