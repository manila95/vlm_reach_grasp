"""RoboCasa adapter (run inside the `robocasa` env: source /scratch/kaustubh/vlm_reach_grasp/robocasa/env.sh).

Conventions here: images are UPRIGHT (RoboCasa datasets/gym are upright); world frame for positions;
the arm action is a delta in the robot BASE frame (env action order: pos3 rot3 grip1 base3 torso1 mode1).
"""
import numpy as np
from robosuite.utils import camera_utils as CU
from robosuite.utils.transform_utils import quat2mat

TASKS = ["PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "PickPlaceDrawerToCounter",
         "PickPlaceSinkToCounter", "PickPlaceToasterToCounter"]
CAMS = ("robot0_agentview_left", "robot0_agentview_right")
SEED0 = {"pretrain": 1111111, "target": 2222222}


def make(task, split, seed, res=256):
    import gymnasium as gym, robocasa  # noqa: F401
    genv = gym.make(f"robocasa/{task}", split=split, seed=seed, camera_depths=True,
                    camera_widths=res, camera_heights=res, disable_env_checker=True)
    return genv


def episode_seed(split, ep):
    return SEED0[split] + ep


def raw(genv):
    return genv.unwrapped.env


def render_rgbd(env, cam, res=512):
    rgb, d = env.sim.render(width=res, height=res, camera_name=cam, depth=True)
    depth = CU.get_real_depth_map(env.sim, d)
    return rgb[::-1].copy(), depth[::-1].copy(), CU.get_camera_intrinsic_matrix(env.sim, cam, res, res), CU.get_camera_extrinsic_matrix(env.sim, cam)


def gt_objects(env):
    return {n: env.sim.data.body_xpos[env.obj_body_id[n]].copy() for n in env.objects}


def scene_info(env):
    ep = env.get_ep_meta()
    cats = {c["name"]: c["info"]["cat"] for c in ep["object_cfgs"]}
    fix = {k: (v[0] if isinstance(v, tuple) else v) for k, v in env.fixture_refs.items()}
    return dict(lang=ep["lang"], target_lang=env.get_obj_lang("obj"), cats=cats,
                fixtures={k: (f.name, np.array(f.pos, float)) for k, f in fix.items()},
                layout=ep.get("layout_id"), style=ep.get("style_id"))


def base_pose(obs):
    """Robot base position and rotation matrix (world) from raw robosuite obs (quats are xyzw)."""
    return obs["robot0_base_pos"].copy(), quat2mat(obs["robot0_base_quat"])


def raw_obs(env):
    return env._get_observations(force_update=True)
