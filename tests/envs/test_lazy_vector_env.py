#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gymnasium as gym

from lerobot.envs.utils import _LazyAsyncVectorEnv


def test_lazy_sync_vector_env_defers_construction_when_spaces_cached():
    created = {"n": 0}

    def _make_env():
        created["n"] += 1
        return gym.make("CartPole-v1")

    probe = gym.make("CartPole-v1")
    lazy = _LazyAsyncVectorEnv(
        [_make_env],
        observation_space=probe.observation_space,
        action_space=probe.action_space,
        metadata=probe.metadata,
        vec_env_cls=gym.vector.SyncVectorEnv,
    )
    probe.close()

    assert created["n"] == 0
    assert lazy._env is None

    obs, _info = lazy.reset()
    assert created["n"] == 1
    assert lazy._env is not None
    assert obs.shape[0] == 1

    lazy.close()
    assert lazy._env is None


def test_lazy_sync_vector_env_probes_once_without_cached_spaces():
    created = {"n": 0}

    def _make_env():
        created["n"] += 1
        return gym.make("CartPole-v1")

    lazy = _LazyAsyncVectorEnv([_make_env], vec_env_cls=gym.vector.SyncVectorEnv)
    # Space probing constructs and closes one temporary env.
    assert created["n"] == 1
    assert lazy._env is None

    lazy.reset()
    assert created["n"] == 2
    lazy.close()
