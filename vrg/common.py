"""Shared helpers: LIBERO env construction, task/object naming."""
import os, glob, re
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data/libero_raw/libero_object")
SUITE = "libero_object"


def get_suite():
    return benchmark.get_benchmark_dict()[SUITE]()


def target_name(language):
    """'pick up the alphabet soup and place it in the basket' -> 'alphabet_soup'. Also accepts a task name with
    underscores and trailing perturbation tags (LIBERO-Plus: 'pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_1')."""
    m = re.match(r"pick[ _]up[ _]the[ _](.+?)[ _]and[ _]place[ _]it[ _]in[ _]the[ _]basket", language)
    return m.group(1).replace(" ", "_")


def make_env(task_id, res=128, depth=False, cams=("agentview", "robot0_eye_in_hand")):
    suite = get_suite()
    t = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res,
                             camera_depths=depth, camera_names=list(cams))
    return env, t


def demo_file(language):
    return os.path.join(RAW, language.replace(" ", "_") + "_demo.hdf5")


# Optional: stronger OSC null-space posture regulation (robosuite default joint_kp=10). Applies to every method.
_NKP = os.environ.get("NULLSPACE_KP")
if _NKP:
    import robosuite.controllers.osc as _osc
    from robosuite.utils.control_utils import nullspace_torques as _nt
    _osc.nullspace_torques = lambda *a, **k: _nt(*a, joint_kp=float(_NKP))
