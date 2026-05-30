# vln/benchmark — 推理测速 (Task 12)

对比两种 frame-context 策略在 VLN per-step 推理上的延迟，量化 KV-cache 复用收益。

## 测什么

两个 setting **共用同一个 NaVIDA prompt builder**（内联在脚本里，镜像
`vln/eval_vllm_navida.py` / `vln/rl/messages.py`，自包含不依赖 vln.rl），
唯一差异是「图片是否跨 step 稳定」，从而隔离出 **KV-cache 复用**这一个变量：

| Setting | 策略 | x 轴 | cache 行为 |
|---|---|---|---|
| 1 `resample` | 每步重新均匀采样 n 帧，图片每次都不同（NaVIDA 式） | n ∈ {2,4,8,16,32,64} | 多模态 prefix cache 必 miss → 每步全量 prefill n 张图 |
| 2 `append` | 历史帧只往后加、前面 bytes 不变（流式） | 历史长度 k | 历史部分 prefix 命中 → 每步只 prefill 新帧（O(1)，与 k 无关）|

- Setting 1：每个 n 跑 `--s1-trials`（默认 100）次，每次全新随机帧，取平均。
- Setting 2：用 `--s2-episodes`（默认 10）个**独立 episode**，每个 episode 自带一套唯一随机帧池，
  内部 k=1..H 顺序发送（episode 内复用 prefix，跨 episode 不污染）；每个 k 在 R 个 episode 上取平均。
- 为隔离 prefill：默认 `ignore_eos=True` + 固定 `--max-tokens`，decode 步数恒定，
  两 setting 延迟差即反映 prefill 成本（策略真正不同的地方）。
- 随机噪声帧与真实帧的 image-token 数相同（只由分辨率决定，默认 640×480 = 真实 habitat 帧），测时延等价。

> 注：NaVIDA 格式里 `current` 帧前有 bridge 文本，每步重算，所以 setting 2 的 floor ≈ 2 张图的 prefill，
> 但**与历史长度无关**——这就是 O(1) vs setting 1 的 O(n) 的对比点。

## 启动 (env1 A100)

```bash
# 终端1：起 vLLM（务必开 prefix caching）
cd /var/data0/sandbox/janec/WorldModel/VLN
bash scripts/vllm_speedtest.sh                 # GPU 0, TP=1, port 8001

# 终端2：跑测速（conda env vln）
cd /var/data0/sandbox/janec/WorldModel/VLN
export OPENAI_API_BASE=http://localhost:8001/v1
export OPENAI_API_KEY=EMPTY
PYTHONPATH=. python -m vln.benchmark.speed_benchmark \
    --out-dir results/speed_bench
```

### 端到端模式（拆 prefill / decode）

默认模式固定 decode=16 token 隔离 prefill。要测**真实端到端延迟**并拆出
TTFT(≈prefill) 与 decode，加 `--stream --no-ignore-eos --max-tokens 512`：

```bash
PYTHONPATH=. python -m vln.benchmark.speed_benchmark \
    --stream --no-ignore-eos --max-tokens 512 \
    --out-dir results/speed_bench_e2e
```

每行会打印 `total / ttft / decode / ct`，results.json 里多出
`ttft_mean_s / ttft_p90_s / decode_mean_s`。

产物（`--out-dir`）：
- `results.json`：全部原始延迟 + 聚合（mean/std/p50/p90 + prompt/completion tokens）
- `setting1_resample.png`：latency vs n
- `setting2_append.png`：latency vs 历史长度 k
- `comparison.png`：两条曲线叠加（x=prompt 内帧数）

## 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--setting` | both | `1` / `2` / `both` |
| `--n-list` | 2,4,8,16,32,64 | setting 1 的 n |
| `--s1-trials` | 100 | setting 1 每个 n 重复次数 |
| `--s2-max-history` | 64 | setting 2 历史长度上限 H |
| `--s2-episodes` | 10 | setting 2 独立 episode 数（= 每个 k 的样本数）|
| `--max-tokens` | 16 | 固定 decode 步数 |
| `--no-ignore-eos` | (off) | 关掉固定输出长度（让模型自然停）|
| `--width/--height` | 640/480 | 帧分辨率（决定 image-token 数）|
| `--plot-only PATH` | — | 只用已有 results.json 重画图 |

## 对照实验（验证 cache 是收益来源）

用 `scripts/vllm_speedtest.sh` 加 `--no-enable-prefix-caching` 重起服务再跑 setting 2，
曲线应退化成随 k 线性增长（≈ setting 1），证明 setting 2 的平坦曲线确实来自 prefix cache。

## Limitation

- 串行发送单请求测 per-step 延迟，不含并发 batching 收益（那是 throughput，另说）。
- 端到端 wall time 含极小 localhost 网络开销；用 `usage` 里 token 数交叉核对 prefill 规模。
- 跨 run 不保证逐毫秒复现（vLLM continuous batching 数值/调度非确定）；看趋势与量级。
- vLLM 0.16 多模态 prefix caching 按帧 bytes 哈希；setting 2 依赖同一帧重编码出 byte-identical JPEG（PIL 确定性编码，已满足）。
