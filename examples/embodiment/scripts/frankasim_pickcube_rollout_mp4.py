# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Roll out FrankaSim PandaPickCube (state) with a trained MLP policy and write MP4."""

from __future__ import annotations

import argparse

import franka_sim  # noqa: F401 — registers gym.envs
import gym
import imageio.v2 as imageio
import numpy as np
import torch
from franka_sim.mujoco_gym_env import GymRenderingSpec

from rlinf.envs.action_utils import prepare_actions_for_mujoco
from rlinf.envs.frankasim.frankasim_env import extract_serl_state
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy


def _strip_checkpoint_prefix(state_dict: dict) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        out[k] = v
    return out


def _infer_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    w0 = state_dict.get("backbone.0.weight")
    if w0 is None:
        raise KeyError("Checkpoint missing backbone.0.weight; cannot infer obs_dim.")
    wm = state_dict.get("actor_mean.weight")
    if wm is None:
        raise KeyError("Checkpoint missing actor_mean.weight; cannot infer action_dim.")
    obs_dim = int(w0.shape[1])
    action_dim = int(wm.shape[0])
    return obs_dim, action_dim


def _reset_obs(env: gym.Env, seed: int | None):
    out = env.reset(seed=seed)
    if isinstance(out, tuple):
        obs, _info = out
        return obs
    return out


def _policy_action_for_panda_pick(
    chunk_actions: torch.Tensor,
    env_action_dim: int,
    model_type: str = "mlp_policy",
) -> np.ndarray:
    """Map policy output to PandaPickCube-v0 actions.

    Must match training: FrankaSim uses ``prepare_actions_for_mujoco``, which
    for 8-D SERL-style vectors takes xyz from indices 0:3 and grip from index 6
    (not a naive ``[..., :4]`` slice).
    """
    raw = chunk_actions.detach().cpu().numpy()
    mapped = prepare_actions_for_mujoco(raw, model_type=model_type)
    out = np.asarray(mapped[0, 0], dtype=np.float32).reshape(-1)
    if out.shape[0] != env_action_dim:
        if out.shape[0] > env_action_dim:
            return out[:env_action_dim]
        pad = np.zeros((env_action_dim,), dtype=np.float32)
        pad[: out.shape[0]] = out
        return pad
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Encode PandaPickCube-v0 rollouts to MP4 using MLPPolicy weights "
            "(FSDP full_weights.pt)."
        )
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Path to actor/model_state_dict/full_weights.pt. Empty: random init.",
    )
    p.add_argument("--out", type=str, required=True, help="Output .mp4 path.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state-key", type=str, default="state")
    p.add_argument("--camera", choices=("front", "wrist"), default="front")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Offscreen render width (franka_sim default is 128; raise for clearer MP4).",
    )
    p.add_argument(
        "--height",
        type=int,
        default=720,
        help="Offscreen render height.",
    )
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )

    render_spec = GymRenderingSpec(width=int(args.width), height=int(args.height))
    env = gym.make(
        "PandaPickCube-v0",
        render_mode="rgb_array",
        render_spec=render_spec,
    )
    env_action_dim = int(np.prod(env.action_space.shape))
    md = getattr(env.unwrapped, "metadata", {})
    fps = int(md.get("render_fps", 50))
    max_episode_steps = int(getattr(env.spec, "max_episode_steps", 100) or 100)

    if args.ckpt:
        raw_sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state_dict = _strip_checkpoint_prefix(raw_sd)
        obs_dim, policy_action_dim = _infer_dims(state_dict)
    else:
        state_dict = None
        obs_dim, policy_action_dim = 10, 8

    model = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=policy_action_dim,
        num_action_chunks=1,
        add_value_head=True,
        add_q_head=False,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    frames: list[np.ndarray] = []

    for ep in range(args.episodes):
        obs = _reset_obs(env, seed=args.seed + ep)
        for _ in range(max_episode_steps + 50):
            frame = env.unwrapped.render()[args.camera]  # type: ignore[index]
            frames.append(np.asarray(frame, dtype=np.uint8))

            st = extract_serl_state(obs, state_key=args.state_key)
            if st.shape[-1] != obs_dim:
                raise ValueError(
                    f"Flat state dim {st.shape[-1]} != model obs_dim {obs_dim}. "
                    "Check --state-key and checkpoint."
                )
            env_obs = {
                "states": torch.from_numpy(st.astype(np.float32)).view(1, -1).to(device)
            }

            chunk_actions, _ = model.predict_action_batch(
                env_obs,
                calculate_logprobs=False,
                calculate_values=False,
                mode="eval",
            )
            a = _policy_action_for_panda_pick(chunk_actions, env_action_dim)
            obs, _rew, terminated, truncated, _info = env.step(a)
            if terminated or truncated:
                break

    env.close()
    imageio.mimsave(args.out, frames, fps=max(fps, 1))
    print(
        f"Wrote {len(frames)} frames to {args.out} "
        f"({args.width}x{args.height} @ {fps} fps)"
    )


if __name__ == "__main__":
    main()
