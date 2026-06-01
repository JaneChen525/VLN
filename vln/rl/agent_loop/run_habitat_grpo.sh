#!/usr/bin/env bash
# V2.0 habitat GRPO launch (verl-native agent_loop). Run inside the verl-dev
# container. env_server must be reachable at VLN_ENV_SERVER_URL.
#
# verl owns: rollout (AgentLoopManager drives HabitatAgentLoop), GRPO advantage,
# actor FSDP update, ref/KL, colocated vLLM + NCCL weight sync, Ray GPU placement.
# We own: HabitatAgentLoop (habitat_loop.py) + this config.
set -xeuo pipefail

WORLDMODEL=${WORLDMODEL:-/workspace/WorldModel}
export PYTHONPATH=${WORLDMODEL}:${PYTHONPATH:-}
export VLN_ENV_SERVER_URL=${VLN_ENV_SERVER_URL:-http://127.0.0.1:8002}

MODEL_PATH=${MODEL_PATH:-${WORLDMODEL}/checkpoints/Qwen3-VL-4B-vln-r2r-merged}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/habitat/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/habitat/train.parquet}
AGENT_CFG=${AGENT_CFG:-${WORLDMODEL}/vln/rl/agent_loop/habitat_agent.yaml}

NGPUS=${NGPUS:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
ROLLOUT_N=${ROLLOUT_N:-4}          # pass_k trials per episode = GRPO group size
ROLLOUT_TP=${ROLLOUT_TP:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
TEMPERATURE=${TEMPERATURE:-0.7}    # T>0 required: T=0 -> identical pass_k -> degenerate GRPO group
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${NGPUS} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CFG}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=habitat \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger=[console] \
    trainer.project_name=verl_v2_habitat \
    trainer.experiment_name=habitat_132_skeleton \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    "$@"
