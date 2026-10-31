"""LeRobot policy server for LIBERO-Plus (runs in LeRobot's own venv: $S/src/lerobot/.venv; no vrg imports).

Reproduces lerobot_eval's per-step chain exactly: raw LIBERO obs (as LiberoEnv._format_raw_obs) -> preprocess_observation
-> task -> env_preprocessor (LiberoProcessorStep: 180-degree image flip, eef pos + axis-angle + gripper qpos)
-> policy preprocessor -> policy.select_action -> postprocessor -> env_postprocessor. The client (vrg.lplus.run
--method lerobot) keeps the simulation in the liberoplus env and sends raw obs over a localhost socket.
  $LRPY vrg/lplus/lerobot_server.py --policy HuggingFaceVLA/smolvla_libero --port 8010
  $LRPY vrg/lplus/lerobot_server.py --policy allenai/MolmoAct2-LIBERO-LeRobot --port 8011 \
      --cams '{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
      --policy_kw '{"inference_action_mode":"continuous","model_dtype":"bfloat16","use_amp":true}'
Messages: {"cmd": "reset"} -> "ok";  {"cmd": "act", "obs": raw_obs, "task": str} -> np.ndarray (7,).
"""
import argparse, json
from multiprocessing.connection import Listener
import numpy as np
import torch

AUTH = b"vrg-lplus"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", required=True); ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--cams", default='{"agentview_image":"image","robot0_eye_in_hand_image":"image2"}')
    ap.add_argument("--policy_kw", default="{}"); ap.add_argument("--size", type=int, default=360); ap.add_argument("--device", default="cuda")
    A = ap.parse_args()
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs import make_env_pre_post_processors, preprocess_observation
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.processor import policy_output_to_transition, transition_to_policy_action
    from lerobot.utils.constants import ACTION
    cams = json.loads(A.cams)
    env_cfg = LiberoEnv(task="libero_object", camera_name=",".join(cams), camera_name_mapping=cams, is_libero_plus=True,
                        observation_height=A.size, observation_width=A.size)
    pcfg = PreTrainedConfig.from_pretrained(A.policy); pcfg.pretrained_path = A.policy; pcfg.device = A.device
    for k, v in json.loads(A.policy_kw).items(): setattr(pcfg, k, v)
    policy = make_policy(cfg=pcfg, env_cfg=env_cfg, rename_map={}); policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=A.policy, preprocessor_overrides={
        "device_processor": {"device": str(policy.config.device)}, "rename_observations_processor": {"rename_map": {}}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pcfg)
    print(f"READY {A.policy} on 127.0.0.1:{A.port}", flush=True)
    with Listener(("127.0.0.1", A.port), authkey=AUTH) as L:
        while True:
            conn = L.accept()
            try:
                while True:
                    m = conn.recv()
                    if m["cmd"] == "reset":
                        policy.reset(); pre.reset(); post.reset(); conn.send("ok"); continue
                    o = m["obs"]
                    raw = {"pixels": {cams[c]: o[c] for c in cams},
                           "robot_state": {"eef": {"pos": o["robot0_eef_pos"], "quat": o["robot0_eef_quat"], "mat": o["eef_mat"]},
                                           "gripper": {"qpos": o["robot0_gripper_qpos"], "qvel": o["robot0_gripper_qvel"]},
                                           "joints": {"pos": o["robot0_joint_pos"], "vel": o["robot0_joint_vel"]}}}
                    raw = {"pixels": {k: v[None] for k, v in raw["pixels"].items()},      # batch of one, as the vector env
                           "robot_state": {g: {k: np.asarray(v, dtype=np.float64)[None] for k, v in d.items()} for g, d in raw["robot_state"].items()}}
                    x = preprocess_observation(raw); x["task"] = [m["task"]]
                    x = pre(env_pre(x))
                    with torch.inference_mode():
                        a = transition_to_policy_action(policy_output_to_transition(policy.select_action(x)))
                    a = env_post({ACTION: post(a)})[ACTION]
                    conn.send(a.to("cpu").float().numpy()[0])
            except (EOFError, ConnectionResetError):
                conn.close()


if __name__ == "__main__":
    main()
