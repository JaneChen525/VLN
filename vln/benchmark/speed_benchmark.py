"""Task 12 — VLN 推理测速：两种 frame-context 策略的 per-step 延迟对比。

两种 setting 共用同一个 NaVIDA prompt builder（vln.rl.messages.build_messages），
唯一区别是「图片是否跨 step 稳定复用」，从而隔离出 KV-cache 复用这一个变量：

  Setting 1 (resample)  每步重新均匀采样 n 帧，图片内容每次都不同 → vLLM 多模态
                        prefix cache 必然 miss → 每次全量 prefill n 张图。
                        x 轴 = n ∈ {2,4,8,16,32,64}，每个 n 跑 --s1-trials 次取平均。

  Setting 2 (append)    历史帧只往后追加、前面帧 bytes 不变 → 历史部分 prefix cache
                        命中 → 每步只 prefill 新帧（NaVIDA 格式里 bridge+current 也会
                        重算，所以 floor ≈ 2 张图，但与历史长度无关，是 O(1) 而非 O(k)）。
                        x 轴 = 历史长度 k，每个 k 用 --s2-episodes 个独立 episode 取平均
                        （每个 episode 自带一套唯一随机帧，避免跨 episode 缓存污染）。

为隔离 prefill 成本，默认 ignore_eos=True + 固定 max_tokens，使 decode 步数恒定，
两个 setting 的延迟差异即反映 prefill（也就是策略真正不同的地方）。

用法见同目录 README.md。
"""

import argparse
import base64
import json
import os
import time
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image

# --- NaVIDA prompt 构造（内联，自包含）---------------------------------------
# 镜像 vln/eval_vllm_navida.py 的 NaVIDA_Agent.act() 与 vln/rl/messages.py 的
# build_messages：内容顺序 [INTRO, *history, BRIDGE, current, tail]。内联是为了让本
# benchmark 不依赖 vln.rl（env1 main checkout 无该模块），可在任意环境独立运行。
# 若上游 prompt 改动，这里需同步。
SYSTEM_PROMPT = "You are a helpful assistant."
PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)
NAVIDA_INTRO = ("Imagine you are a robot programmed for navigation tasks. "
                "You have been given a video of historical observations")
NAVIDA_BRIDGE = "and an image of the current observation"

DEFAULT_INSTRUCTION = (
    "Walk out of the bedroom and turn left. Walk down the hallway and stop "
    "in front of the kitchen table."
)


def encode_image_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def uniform_sample_with_ends(data, n):
    if len(data) <= n:
        return data
    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]


def build_messages(instruction, rgb_history, k_history=8):
    current = rgb_history[-1]
    historic = uniform_sample_with_ends(rgb_history[:-1], k_history) if len(rgb_history) > 1 else [current]
    tail = PROMPT_TEMPLATE.format(instruction).split("current observation")[1]
    content = [{"type": "text", "text": NAVIDA_INTRO}]
    content.extend(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(im)}"}}
        for im in historic
    )
    content.append({"type": "text", "text": NAVIDA_BRIDGE})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(current)}"}})
    content.append({"type": "text", "text": tail})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
# -----------------------------------------------------------------------------


def make_image(rng: np.random.Generator, width: int, height: int) -> Image.Image:
    """随机 RGB 帧。图像内容不影响 Qwen3-VL 的 image-token 数（只由分辨率决定），
    所以用随机噪声测时延与用真实帧等价，但能保证 setting 1 每帧内容不同（cache miss）。"""
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def percentile(latencies, q):
    return float(np.percentile(latencies, q)) if latencies else None


def summarize(latencies, prompt_tokens, completion_tokens, ttfts=None):
    out = {
        "n_samples": len(latencies),
        "mean_s": float(np.mean(latencies)),
        "std_s": float(np.std(latencies)),
        "min_s": float(np.min(latencies)),
        "p50_s": percentile(latencies, 50),
        "p90_s": percentile(latencies, 90),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latencies_s": [float(x) for x in latencies],
    }
    ttfts = [t for t in (ttfts or []) if t is not None]
    if ttfts:
        # TTFT ≈ prefill 成本；decode = total - ttft。端到端拆解用。
        out["ttft_mean_s"] = float(np.mean(ttfts))
        out["ttft_p90_s"] = percentile(ttfts, 90)
        out["decode_mean_s"] = out["mean_s"] - out["ttft_mean_s"]
    return out


def call_once(client, model, messages, max_tokens, ignore_eos, temperature, stream=False):
    """发一次请求。返回 (total_s, ttft_s, prompt_tokens, completion_tokens)。
    stream=True 时用流式抓 TTFT（首 token 时间 ≈ prefill），否则 ttft_s=None。只计时 API 调用本身。"""
    extra_body = {"ignore_eos": True} if ignore_eos else {}
    if not stream:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=max_tokens,
            temperature=temperature, top_p=0.95, extra_body=extra_body,
        )
        dt = time.perf_counter() - t0
        usage = resp.usage
        pt = getattr(usage, "prompt_tokens", None) if usage else None
        ct = getattr(usage, "completion_tokens", None) if usage else None
        return dt, None, pt, ct

    t0 = time.perf_counter()
    ttft, pt, ct = None, None, None
    resp = client.chat.completions.create(
        model=model, messages=messages, max_completion_tokens=max_tokens,
        temperature=temperature, top_p=0.95, extra_body=extra_body,
        stream=True, stream_options={"include_usage": True},
    )
    for chunk in resp:
        if ttft is None and chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            ttft = time.perf_counter() - t0
        if getattr(chunk, "usage", None):
            pt = chunk.usage.prompt_tokens
            ct = chunk.usage.completion_tokens
    total = time.perf_counter() - t0
    return total, ttft, pt, ct


def run_setting1(client, model, args, instruction):
    """每个 n 跑 s1_trials 次，每次全新随机帧（无 cache 复用）。"""
    rng = np.random.default_rng(args.seed)
    n_list = [int(x) for x in args.n_list.split(",")]
    per_n = {}
    for n in n_list:
        lat, ttfts, pt, ct = [], [], None, None
        # warmup（不计时），用与正式请求同结构的全新帧
        for _ in range(args.warmup):
            frames = [make_image(rng, args.width, args.height) for _ in range(n)]
            msgs = build_messages(instruction, frames, k_history=n)
            call_once(client, model, msgs, args.max_tokens, args.ignore_eos, args.temperature, args.stream)
        for _ in range(args.s1_trials):
            frames = [make_image(rng, args.width, args.height) for _ in range(n)]
            msgs = build_messages(instruction, frames, k_history=n)
            dt, tf, p, c = call_once(client, model, msgs, args.max_tokens, args.ignore_eos, args.temperature, args.stream)
            lat.append(dt)
            ttfts.append(tf)
            pt, ct = p, c
        per_n[str(n)] = summarize(lat, pt, ct, ttfts)
        s = per_n[str(n)]
        extra = f"  ttft={s['ttft_mean_s']*1000:7.1f}ms  decode={s['decode_mean_s']*1000:7.1f}ms  ct={ct}" if "ttft_mean_s" in s else f"  prompt_tokens={pt}"
        print(f"[setting1] n={n:>3}  total={s['mean_s']*1000:7.1f}ms{extra}")
    return {"n_list": n_list, "s1_trials": args.s1_trials, "per_n": per_n}


def run_setting2(client, model, args, instruction):
    """R 个独立 episode，每个 episode 内 k=1..H 顺序发送（历史帧 bytes 稳定 → prefix cache 复用）。
    每个 episode 用唯一随机帧池，避免跨 episode 缓存命中污染「增量成本」测量。"""
    H = args.s2_max_history
    R = args.s2_episodes
    # per_k[k] 收集 R 个 episode 在该长度的延迟
    per_k_lat = {k: [] for k in range(1, H + 1)}
    per_k_ttft = {k: [] for k in range(1, H + 1)}
    per_k_tokens = {k: (None, None) for k in range(1, H + 1)}

    # warmup：单独一个 episode 把链路热起来（不计入结果）
    if args.warmup > 0:
        wrng = np.random.default_rng(args.seed + 9991)
        pool = [make_image(wrng, args.width, args.height) for _ in range(min(args.warmup, H))]
        for k in range(1, len(pool) + 1):
            msgs = build_messages(instruction, pool[:k], k_history=k)
            call_once(client, model, msgs, args.max_tokens, args.ignore_eos, args.temperature, args.stream)

    for ep in range(R):
        ep_rng = np.random.default_rng(args.seed + 1000 + ep)
        pool = [make_image(ep_rng, args.width, args.height) for _ in range(H)]
        for k in range(1, H + 1):
            msgs = build_messages(instruction, pool[:k], k_history=k)
            dt, tf, p, c = call_once(client, model, msgs, args.max_tokens, args.ignore_eos, args.temperature, args.stream)
            per_k_lat[k].append(dt)
            per_k_ttft[k].append(tf)
            per_k_tokens[k] = (p, c)
        print(f"[setting2] episode {ep+1}/{R} done (H={H})")

    per_k = {}
    for k in range(1, H + 1):
        pt, ct = per_k_tokens[k]
        per_k[str(k)] = summarize(per_k_lat[k], pt, ct, per_k_ttft[k])
    # 简报：抽几个 k 打印
    for k in sorted({1, 2, 4, 8, 16, 32, H} & set(range(1, H + 1))):
        s = per_k[str(k)]
        extra = f"  ttft={s['ttft_mean_s']*1000:7.1f}ms  decode={s['decode_mean_s']*1000:7.1f}ms  ct={s['completion_tokens']}" if "ttft_mean_s" in s else f"  prompt_tokens={s['prompt_tokens']}"
        print(f"[setting2] k={k:>3}  total={s['mean_s']*1000:7.1f}ms{extra}")
    return {"max_history": H, "episodes": R, "per_k": per_k}


def plot_results(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s1 = results.get("setting1")
    s2 = results.get("setting2")

    # fig1: setting1 latency vs n
    if s1:
        ns = sorted(int(k) for k in s1["per_n"])
        means = [s1["per_n"][str(n)]["mean_s"] * 1000 for n in ns]
        stds = [s1["per_n"][str(n)]["std_s"] * 1000 for n in ns]
        plt.figure(figsize=(7, 5))
        plt.errorbar(ns, means, yerr=stds, marker="o", capsize=4)
        plt.xscale("log", base=2)
        plt.xticks(ns, [str(n) for n in ns])
        plt.xlabel("n (uniformly-sampled frames, resampled each step)")
        plt.ylabel("per-step latency (ms)")
        plt.title("Setting 1: resample n frames (no KV-cache reuse)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "setting1_resample.png"), dpi=130)
        plt.close()

    # fig2: setting2 latency vs history length k
    if s2:
        ks = sorted(int(k) for k in s2["per_k"])
        means = [s2["per_k"][str(k)]["mean_s"] * 1000 for k in ks]
        stds = [s2["per_k"][str(k)]["std_s"] * 1000 for k in ks]
        plt.figure(figsize=(7, 5))
        plt.errorbar(ks, means, yerr=stds, marker=".", capsize=2)
        plt.xlabel("history length k (append-only, KV-cache reuse)")
        plt.ylabel("per-step latency (ms)")
        plt.title("Setting 2: append history frame (KV-cache reuse)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "setting2_append.png"), dpi=130)
        plt.close()

    # fig3: overlay — latency vs #frames in prompt
    if s1 and s2:
        ns = sorted(int(k) for k in s1["per_n"])
        s1_means = [s1["per_n"][str(n)]["mean_s"] * 1000 for n in ns]
        ks = sorted(int(k) for k in s2["per_k"])
        s2_means = [s2["per_k"][str(k)]["mean_s"] * 1000 for k in ks]
        plt.figure(figsize=(8, 5))
        plt.plot(ns, s1_means, marker="o", label="Setting 1 resample (cold prefill)")
        plt.plot(ks, s2_means, marker=".", label="Setting 2 append (cached prefix)")
        plt.xlabel("# image frames in prompt")
        plt.ylabel("per-step latency (ms)")
        plt.title("Resample vs append: KV-cache reuse savings")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "comparison.png"), dpi=130)
        plt.close()
    print(f"[plot] PNGs written to {out_dir}")


def build_client(args):
    from openai import OpenAI
    base_url = args.base_url or os.environ.get("OPENAI_API_BASE")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    assert base_url, "need --base-url or OPENAI_API_BASE"
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = args.model or client.models.list().data[0].id
    return client, model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=None, help="vLLM OpenAI endpoint, e.g. http://localhost:8001/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=None, help="served model id；缺省自动取 /v1/models 第一个")
    ap.add_argument("--setting", choices=["1", "2", "both"], default="both")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--max-tokens", type=int, default=16, help="固定 decode 步数（配 ignore_eos）以隔离 prefill")
    ap.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    ap.set_defaults(ignore_eos=True)
    ap.add_argument("--stream", action="store_true",
                    help="流式请求，抓 TTFT(≈prefill) 与 total，拆解端到端延迟（端到端模式建议配 --no-ignore-eos --max-tokens 512）")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    # setting 1
    ap.add_argument("--n-list", default="2,4,8,16,32,64")
    ap.add_argument("--s1-trials", type=int, default=100)
    # setting 2
    ap.add_argument("--s2-max-history", type=int, default=64)
    ap.add_argument("--s2-episodes", type=int, default=10)
    # io
    ap.add_argument("--out-dir", default="results/speed_bench")
    ap.add_argument("--plot-only", default=None, help="只重画图：传入已有 results.json 路径")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.plot_only:
        with open(args.plot_only) as f:
            results = json.load(f)
        plot_results(results, args.out_dir)
        return

    instruction = args.instruction

    client, model = build_client(args)
    print(f"[init] model={model} base_url={args.base_url or os.environ.get('OPENAI_API_BASE')} "
          f"img={args.width}x{args.height} max_tokens={args.max_tokens} ignore_eos={args.ignore_eos}")

    results = {
        "meta": {
            "model": model,
            "base_url": args.base_url or os.environ.get("OPENAI_API_BASE"),
            "width": args.width,
            "height": args.height,
            "max_tokens": args.max_tokens,
            "ignore_eos": args.ignore_eos,
            "stream": args.stream,
            "temperature": args.temperature,
            "warmup": args.warmup,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    }

    if args.setting in ("1", "both"):
        results["setting1"] = run_setting1(client, model, args, instruction)
    if args.setting in ("2", "both"):
        results["setting2"] = run_setting2(client, model, args, instruction)

    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[done] results -> {out_json}")
    plot_results(results, args.out_dir)


if __name__ == "__main__":
    main()
