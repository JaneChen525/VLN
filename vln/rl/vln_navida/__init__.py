"""VLN full-episode online GRPO for verl (design: report/016).

Full-episode rollout -> flatten every NaVIDA decision into an independent
training sample -> trajectory-level GRPO. Reuses env_server (HTTP) + messages.py.
"""
