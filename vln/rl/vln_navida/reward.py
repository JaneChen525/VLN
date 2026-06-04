"""Trajectory reward (016 §11). Stage 1 = sparse outcome (success).

Landing order: success -> success + progress -> success + spl + progress - penalties.
Keep shaping weights small so GRPO signal stays tied to the nav goal.
"""


def compute_trajectory_reward(metrics: dict, progress_coef: float = 0.0) -> float:
    """metrics from VLNEnv.metrics(): success/spl/ne/oracle_success/collisions/...

    Stage 1 default: reward = success (1.0/0.0). Set progress_coef>0 to add a small
    oracle-success progress shaping when success is too sparse.
    """
    reward = float(metrics.get("success", 0.0))
    if progress_coef:
        reward += progress_coef * float(metrics.get("oracle_success", 0.0))
    return reward
