"""13.4 Rollout-only smoke: drive full episodes via env_server + a standalone vLLM
OpenAI endpoint (same inference path as eval_vllm_navida.py), record traces, report SR.

No verl, no training. Validates VLNEnv + NaVIDA prompt + action parser + run_episode
against the existing eval behavior before wiring into verl (13.5).

  python -m vln.rl.vln_navida.rollout_smoke \
    --env-url http://127.0.0.1:8002 --vllm-url http://127.0.0.1:8000 \
    --model <served-name> --n-episodes 5 --temperature 0.0 --out /tmp/vln_rollout_trace.json
"""
import argparse
import asyncio
import json

import httpx

from vln.rl.vln_navida.env_pool import VLNEnv
from vln.rl.vln_navida.full_episode_agent_loop import DecisionGen, run_episode
from vln.rl.vln_navida.prompt import build_navida_messages_b64


def make_decide(client: httpx.AsyncClient, model: str, temperature: float, top_p: float, max_tokens: int):
    async def decide(instruction, frame_buffer) -> DecisionGen:
        # forward env_server JPEG b64 directly (no decode/re-encode) == V1 + eval
        messages, sel = build_navida_messages_b64(instruction, frame_buffer, k_history=8)
        r = await client.post("/v1/chat/completions", json={
            "model": model, "messages": messages,
            "temperature": temperature, "top_p": top_p, "max_completion_tokens": max_tokens,
        })
        r.raise_for_status()
        return DecisionGen(action_text=r.json()["choices"][0]["message"]["content"], selected_frame_ids=sel)
    return decide


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-url", default="http://127.0.0.1:8002")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--max-env-steps", type=int, default=400)       # eval max_episode_steps
    ap.add_argument("--max-action-history", type=int, default=200)  # eval_vllm.sh value
    ap.add_argument("--temperature", type=float, default=0.3)  # eval default
    ap.add_argument("--top-p", type=float, default=0.95)        # eval default
    ap.add_argument("--max-tokens", type=int, default=512)      # eval default
    ap.add_argument("--concurrency", type=int, default=1)  # parallel episodes (<= env_server pool)
    ap.add_argument("--out", default="/tmp/vln_rollout_trace.json")
    args = ap.parse_args()

    vclient = httpx.AsyncClient(base_url=args.vllm_url, timeout=120.0)
    decide = make_decide(vclient, args.model, args.temperature, args.top_p, args.max_tokens)

    async with httpx.AsyncClient(base_url=args.env_url, timeout=120.0) as ec:
        episode_ids = (await ec.get("/episodes", params={"limit": args.n_episodes})).json()["episode_ids"]

    sem = asyncio.Semaphore(args.concurrency)

    async def one(ep):
        async with sem:
            env = VLNEnv(args.env_url)
            try:
                traj = await run_episode(
                    env, {"episode_id": ep}, decide,
                    group_uid=ep, trajectory_uid=f"{ep}#0",
                    max_decisions=args.max_decisions, max_env_steps=args.max_env_steps,
                    max_action_history=args.max_action_history,
                )
            finally:
                await env.close()
            print(f"ep={ep} reward={traj.reward} decisions={traj.metrics['num_decisions']} "
                  f"steps={traj.metrics['env_steps']} success={traj.metrics['success']} "
                  f"ne={traj.metrics['ne']:.2f}", flush=True)
            return traj

    trajs = await asyncio.gather(*[one(ep) for ep in episode_ids[: args.n_episodes]])
    await vclient.aclose()
    sr = sum(t.reward for t in trajs) / max(1, len(trajs))
    print(f"\n=== {len(trajs)} episodes | SR={sr:.3f} ===")

    with open(args.out, "w") as f:
        json.dump([{
            "episode_id": t.episode_id, "reward": t.reward, "metrics": t.metrics,
            "decisions": [{"turn_id": d.turn_id, "action_text": d.action_text,
                           "parsed": d.parsed_actions, "atomic": d.atomic_chunk} for d in t.decisions],
        } for t in trajs], f, indent=2)
    print(f"trace -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
