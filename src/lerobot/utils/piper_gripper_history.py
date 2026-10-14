"""Causal recorded-action history and matching policy-control-step deployment history.

No video decoding, robot access, or dependency on future/current action targets here.
Recorded ``action`` is only a command if the dataset collector stored commands.
"""

from collections import deque

import numpy as np

HISTORY_KEY = "observation.gripper_history"
CONTEXT_KEY = "observation.piper_state_gripper_context"
HISTORY_STEPS = 20
CONTEXT_DIM = 32


class RecordedGripperHistory:
    """Index complete, selected episodes in their HF dataset row order (not Arrow backing order)."""

    def __init__(self, columns, action_names, state_names):
        episodes = np.asarray(columns["episode_index"], dtype=np.int64)
        frames = np.asarray(columns["frame_index"], dtype=np.int64)
        actions = np.asarray(columns["action"], dtype=np.float32)
        states = np.asarray(columns["observation.state"], dtype=np.float32)
        if not len(episodes) or actions.shape != (len(episodes), 7) or states.shape != actions.shape:
            raise ValueError("Gripper history needs nonempty seven-dimensional Piper rows")
        self.commands = actions[:, action_names.index("gripper.pos")].copy()
        self.initial_states = states[:, state_names.index("gripper.pos")].copy()
        if not np.isfinite(self.commands).all() or not np.isfinite(self.initial_states).all():
            raise ValueError("Nonfinite gripper history")
        self.starts = np.empty(len(episodes), dtype=np.int64)
        seen = set()
        start = 0
        for i, episode in enumerate(episodes):
            if i == 0 or episode != episodes[i - 1]:
                if int(episode) in seen:
                    raise ValueError("History requires contiguous complete episodes")
                seen.add(int(episode))
                start = i
            if frames[i] != i - start:
                raise ValueError("History requires complete episodes starting at frame 0, without gaps")
            self.starts[i] = start

    def __getitem__(self, index):
        start = self.starts[index]
        # Left padding is the INITIAL measured state, never action[0] (a target at t=0).
        result = np.full(HISTORY_STEPS, self.initial_states[start], dtype=np.float32)
        past = self.commands[max(start, index - HISTORY_STEPS) : index]
        if len(past):
            result[-len(past) :] = past
        return result


def configure_gripper_history(dataset, policy_config):
    """Opt in after episode selection; cache only small numeric columns, never video frames."""
    if not getattr(policy_config, "piper_gripper_history_steps", 0):
        return dataset
    from lerobot.utils.piper_training_eval import piper_names

    names, state_names = piper_names(dataset.meta.features)
    keys = ["episode_index", "frame_index", "action", "observation.state"]
    columns = dataset.hf_dataset.select_columns(keys).with_format(None)[:]
    dataset._piper_gripper_history = RecordedGripperHistory(columns, names, state_names)
    return dataset


class ExecutedGripperHistory:
    """One entry per successfully submitted POLICY control step; exclude 100 Hz keepalives.

    Reset by constructing one per episode. These are commanded openings, not feedback
    or an assertion that the robot physically reached the target. Inference pauses do
    not advance the buffer, so 20 steps need not be exactly two wall-clock seconds.
    """

    def __init__(self, initial_gripper):
        if not np.isfinite(initial_gripper):
            raise ValueError("Nonfinite initial gripper state")
        self.values = deque([float(initial_gripper)] * HISTORY_STEPS, maxlen=HISTORY_STEPS)

    def append(self, command):
        if not np.isfinite(command):
            raise ValueError("Nonfinite gripper command")
        # Match PiperConnectionMixin.send_action's normalized opening -> integer command conversion.
        self.values.append(round(max(0.0, min(1.0, float(command))) * 70000) / 70000)

    def observation(self):
        return np.asarray(self.values, dtype=np.float32).copy()
