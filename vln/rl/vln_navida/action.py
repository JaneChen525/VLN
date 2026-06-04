"""NaVIDA text-action parsing + Habitat atomic-action expansion (016 §13).

Thin reuse of vln/rl/messages.py so rollout parsing == eval parsing. NaVIDA emits
up to ACTIONS_PER_TURN sub-actions per decision (eval select_action_idx=2); each
sub-action expands to atomic Habitat steps (0=stop,1=fwd,2=left,3=right).
"""
from vln.rl.messages import expand_to_atomic, parse_actions

FORWARD_DISTANCE = 25  # cm per forward
TURN_ANGLE = 15        # deg per turn
ACTIONS_PER_TURN = 2   # eval select_action_idx=2


def parse_navida_action(text: str, n: int = ACTIONS_PER_TURN):
    """Model text -> list of up to n (action_id, numeric) sub-actions."""
    return parse_actions(text, FORWARD_DISTANCE, TURN_ANGLE, n)


def to_atomic_chunk(parsed: list) -> list[int]:
    """Parsed sub-actions -> flat atomic Habitat action list (one per env.step)."""
    chunk: list[int] = []
    for action_id, numeric in parsed:
        if action_id is None:
            continue
        chunk.extend(expand_to_atomic(action_id, numeric, FORWARD_DISTANCE, TURN_ANGLE))
    return chunk
