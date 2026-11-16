"""
Records everything Lift produces over seeded random rollouts into an .npz, so two code versions can be compared
exactly. Used by check_lift_unchanged.py; can also be run on its own.

Recorded at every step: all observations (RGB, depth, proprio, object state), reward, done, _check_success() and the
full sim state (qpos, qvel).

Usage:
    MUJOCO_GL=egl python tests/ycb/lift_record.py out.npz [--dataset path/to/lift/image_abs.hdf5]

With --dataset, also records a rollout of the env built by robomimic from the dataset metadata, the way
diffusion_policy's RobomimicImageRunner builds it for eval.
"""
import argparse

import numpy as np
import robosuite as suite

# (tag, object_type, camera_depths, hard_reset)
CONFIGS = [
    ("box", "box", False, True),
    ("cylinder", "cylinder", False, True),
    ("can_xml", "can", False, True),
    ("lemon_xml", "lemon", False, True),
    ("banana", "011_banana", False, True),
    ("banana_depth", "011_banana", True, True),
    ("banana_softreset", "011_banana", False, False),
    ("cracker_box", "003_cracker_box", True, True),
    ("mug", "025_mug", False, True),
]
N_EPISODES = 3
N_STEPS = 30


def record_lift(rec):
    for tag, obj, depth, hard_reset in CONFIGS:
        np.random.seed(0)
        env = suite.make(
            "Lift", robots="Panda", object_type=obj, has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=True, use_object_obs=True, reward_shaping=True, hard_reset=hard_reset,
            camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=84, camera_widths=84,
            camera_depths=depth,
        )
        for ep in range(N_EPISODES):
            obs = env.reset()
            for t in range(N_STEPS):
                for k, v in obs.items():
                    rec[f"{tag}/ep{ep}/t{t}/obs/{k}"] = np.asarray(v)
                rec[f"{tag}/ep{ep}/t{t}/qpos"] = env.sim.data.qpos.copy()
                rec[f"{tag}/ep{ep}/t{t}/qvel"] = env.sim.data.qvel.copy()
                obs, r, done, info = env.step(np.random.uniform(-1, 1, env.action_dim))
                rec[f"{tag}/ep{ep}/t{t}/reward"] = np.array(r)
                rec[f"{tag}/ep{ep}/t{t}/done"] = np.array(done)
                rec[f"{tag}/ep{ep}/t{t}/success"] = np.array(env._check_success())
        env.close()
        print("recorded", tag, flush=True)


def record_robomimic(rec, dataset):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_modality_mapping_from_dict({
        "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
        "low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    })
    # same overrides as RobomimicImageRunner
    meta = FileUtils.get_env_metadata_from_dataset(dataset)
    meta["env_kwargs"]["use_object_obs"] = False
    meta["env_kwargs"]["object_type"] = "011_banana"
    meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    env = EnvUtils.create_env_from_metadata(env_meta=meta, render=False, render_offscreen=True, use_image_obs=True)

    np.random.seed(0)
    obs = env.reset()
    eef0 = obs["robot0_eef_pos"].copy()
    for t in range(N_STEPS):
        for k, v in obs.items():
            rec[f"robomimic/t{t}/obs/{k}"] = np.asarray(v)
        # absolute actions: small random targets around the start pose
        action = np.concatenate([
            eef0 + np.random.uniform(-0.05, 0.05, 3), np.random.uniform(-0.2, 0.2, 3), [np.random.choice([-1, 1])],
        ])
        obs, r, done, info = env.step(action)
        rec[f"robomimic/t{t}/reward"] = np.array(r)
        rec[f"robomimic/t{t}/success"] = np.array(env.is_success()["task"])
    print("recorded robomimic", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", help="output .npz path")
    parser.add_argument("--dataset", default=None, help="Lift robomimic hdf5, to also record the eval env path")
    args = parser.parse_args()

    rec = {}
    record_lift(rec)
    if args.dataset is not None:
        record_robomimic(rec, args.dataset)
    np.savez_compressed(args.out, **rec)
    print("saved", len(rec), "arrays to", args.out)
