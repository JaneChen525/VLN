"""7.3d-0 spike step 1: probe what weight-update mechanisms vLLM exposes.

Run inside the `vln` conda env on env1:
    python -m vln.rl.weight_sync.probe_vllm
Prints: vLLM version, server routes, dev-mode gating, and the in-process
LLM / engine methods relevant to RLHF weight sync (collective_rpc,
update/load weights, sleep/wake).
"""
import inspect
import re


def probe_routes():
    from vllm.entrypoints.openai import api_server as a
    src = inspect.getsource(a)
    routes = sorted(set(re.findall(r"router\.(?:get|post|put)\(\s*['\"]([^'\"]+)", src)))
    print("SERVER ROUTES:", routes)
    print("--- DEV_MODE / weight / collective lines ---")
    for line in src.splitlines():
        s = line.strip()
        if ("DEV_MODE" in s) or ("collective_rpc" in s) or ("weight" in s.lower()):
            print("  ", s[:110])


def probe_llm_api():
    import vllm
    print("\nvLLM version:", vllm.__version__)
    from vllm import LLM
    names = [n for n in dir(LLM) if not n.startswith("__")]
    keys = [n for n in names if any(k in n.lower() for k in
            ["weight", "load", "collective", "rpc", "sleep", "wake", "reset"])]
    print("LLM weight-sync methods:", keys)


def probe_engine_api():
    try:
        from vllm.engine.llm_engine import LLMEngine
        names = [n for n in dir(LLMEngine) if not n.startswith("_")]
        keys = [n for n in names if any(k in n.lower() for k in
                ["weight", "collective", "rpc", "sleep", "wake"])]
        print("LLMEngine weight-sync methods:", keys)
    except Exception as e:
        print("LLMEngine probe failed:", e)


if __name__ == "__main__":
    probe_routes()
    probe_llm_api()
    probe_engine_api()
