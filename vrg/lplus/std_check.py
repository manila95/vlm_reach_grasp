"""Sanity check for vrg/lplus/lerobot_server.py: a LeRobot policy on the 10 standard LIBERO-Object tasks (the LIBERO-Plus
package ships the original bddl/init files). SmolVLA-LIBERO: 17/20 (2 init states per task; published ~90%).
  $LPY vrg/lplus/std_check.py PORT N_INIT WAIT
"""
import glob, os, sys, numpy as np, torch
from multiprocessing.connection import Client
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
port, n_init, wait = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
B = get_libero_path("bddl_files") + "/libero_object"; I = get_libero_path("init_states") + "/libero_object"
names = sorted(os.path.basename(f)[:-5] for f in glob.glob(B + "/*.bddl") if f.endswith("basket.bddl"))
names = [n for n in names if os.path.exists(f"{I}/{n}.init") or os.path.exists(f"{I}/{n}.pruned_init")][:10]
conn = Client(("127.0.0.1", port), authkey=b"vrg-lplus"); tot = []
keys = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel", "robot0_joint_pos", "robot0_joint_vel")
for n in names:
    f = f"{I}/{n}.pruned_init" if os.path.exists(f"{I}/{n}.pruned_init") else f"{I}/{n}.init"
    inits = torch.load(f, weights_only=False); lang = n.replace("_", " ")
    for k in range(n_init):
        env = OffScreenRenderEnv(bddl_file_name=f"{B}/{n}.bddl", camera_heights=360, camera_widths=360)
        env.seed(7); env.reset(); obs = env.set_init_state(inits[k])
        for _ in range(wait): obs, _, done, _ = env.step([0.0] * 6 + [-1.0])
        conn.send({"cmd": "reset"}); conn.recv(); done = False
        for step in range(280):
            o = {q: obs[q] for q in keys}; o["eef_mat"] = env.env.robots[0].controller.ee_ori_mat.copy()
            conn.send({"cmd": "act", "obs": o, "task": lang}); obs, _, done, _ = env.step(np.asarray(conn.recv(), dtype=np.float64).tolist())
            if done: break
        env.close(); tot.append(done); print(n, k, done, flush=True)
print("STD", np.mean(tot), len(tot), flush=True)
