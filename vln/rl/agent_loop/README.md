# agent_loop — V2.0 verl-native RL

`HabitatAgentLoop`：把 habitat VLN rollout 接进 **verl 原生 agent_loop**。verl 接管
rollout 调度 / GRPO advantage / actor FSDP / ref+KL / colocated vLLM NCCL 权重同步 /
Ray GPU placement；本组件只负责「用 LLM 输出驱动 habitat env + 算 reward」。

V2.0 设计与分阶段计划见 [report/015](../../../../report/015-rl-v2-verl-native.md)。
V1.0（解耦 rollout+parquet+手写 orchestrator，已弃用作备份）见 [report/011](../../../../report/011-rl-dev-progress.md)。

## 文件

| File | Role |
|---|---|
| `habitat_loop.py` | `HabitatAgentLoop(AgentLoopBase)` + `build_navida_content`。`run()` 跑一个 episode：reset env_server → 建 NaVIDA prompt（live 帧）→ `server_manager.generate` → `parse_actions` → `env.step` → 返回 `AgentLoopOutput`（tokens + `reward_score`）。|
| `habitat_agent.yaml` | verl agent 注册条目（`name: habitat` + `_target_` + `env_server_url`）。hydra 按 `_target_` 实例化，verl 无需 import 本包。|
| `build_episodes_parquet.py` | 建驱动 verl dataloader 的 parquet：每行一个 episode（占位 prompt + `extra_info.episode_id`，**不存帧**——帧 rollout 时从 env_server 取）。|
| `run_habitat_grpo.sh` | `verl.trainer.main_ppo` 启动脚本（GRPO + colocated vLLM + ref/KL + Ray 自动资源）。|

## 数据流

```
parquet 行(episode_id) ──> verl dataloader ──> HabitatAgentLoop.run()
                                                  │ reset/step/delete (HTTP)
                                                  ▼
                                            env_server (habitat, 独立进程/GPU)
run() 返回 AgentLoopOutput(prompt_ids, response_ids, response_mask,
        multi_modal_data={images}, reward_score=SR)
   └─> verl: GRPO advantage(按 prompt uid 分组=同 episode 的 n trials)
            + actor FSDP update + ref logprob/KL + colocated vLLM NCCL sync
```

- **GRPO 分组**：一行 parquet = 一个 episode = 一个 group；`rollout.n`(=pass_k) 提供组内 trials。group key 不含 trial_id（对齐 [report/007](../../../../report/007-rl-framework.md)）。
- **Reward**：sparse SR（`AgentLoopOutput.reward_score` → verl `rm_scores[-1]`），GRPO 按 response_mask 广播到所有 action token。
- **帧 / prompt**：复用 [`messages.py`](../messages.py) 的 NaVIDA 文本常量 + `parse_actions` + `uniform_sample_with_ends`，与 eval / V1 同源防漂移。`build_navida_content` 把帧表达成 verl 多模态 markers（`{"type":"image"}` + 单独 images 列表）。

## 用法（env1，verl-dev 容器内）

```bash
# 0. 起 env_server（host conda vln，GPU 6，见 ../env_server/README.md）
#    healthz ok:true 后继续
# 1. 建 parquet（容器内）
PYTHONPATH=/workspace/WorldModel python -m vln.rl.agent_loop.build_episodes_parquet \
    --env-server-url http://127.0.0.1:8002 --n 8 --out ~/data/habitat/train.parquet
# 2. 跑 GRPO（默认 1 step skeleton）
bash /workspace/WorldModel/vln/rl/agent_loop/run_habitat_grpo.sh
```

verl-dev 是 `--network host`，容器内 `127.0.0.1:8002` 直达 host env_server。

## 关键配置 / 坑

- **`env_server_url` 走 yaml/默认值，不能只靠环境变量**：verl 的 Ray `runtime_env.env_vars` 是固定白名单，driver 设的 `VLN_ENV_SERVER_URL` 不会传到 Ray worker。默认 `http://127.0.0.1:8002`，可在 `habitat_agent.yaml` 改，或设 `VLN_ENV_SERVER_URL`（仅当能保证 worker 继承时）。
- **`max_model_len=8192`**：Qwen3-VL 默认 max_seq_len=262144，colocated 下 KV cache 装不下（40G A100）。VLN 序列 max≈4096+256，设 8192。
- **`temperature>0`**：T=0 → 同 episode 的 pass_k trials 完全相同 → GRPO 组退化 → advantage=0。默认 0.7。
- **env_server pool < rollout 并发**：`run()` 的 reset 带有界重试（503「no idle worker」时退避等待），pool 小于并发也能排队跑通。
- **KL 必开**（`use_kl_loss=True`）：V1 实测无 KL 一步即让 policy STOP 行为崩溃。verl 原生 ref worker。

## 现状 / limitation

- **13.2（单轮 skeleton，已通）**：`run()` 只做一轮 LLM 调用 + step 解析出的原子动作。1 step GRPO 端到端跑通（env1 4×A100，8 traj，step 68s，peak 20.7GB/卡），token + reward 流通、Ray 自动调度 + colocated NCCL sync 1.07s 验证。
  - **reward=0**：单轮够不到目标，SR 恒 0（管道通，值为 0）。真实多轮 SR 见 13.3。
- **13.3（待做）**：多轮帧累积（NaVIDA 协议跨 turn buffer）+ 真实 episode 跑到 done/max_turns + 非零 SR reward。
- 单模型（无 world model 分支），多变体超参（frame_selection / actions_per_turn）经 [`selectors.py`](../selectors.py) 预留。
