#!/bin/bash
# Demo clips for the notebook: the same episodes under different configs, run one after another (one Isaac process at a time).
R=/scratch/kaustubh/vlm_reach_grasp/isaac; G=$R/ground_eval
E1="103:021_bleach_cleanser,104:006_mustard_bottle,106:008_pudding_box"
$R/run.sh -m vrg.franka.run_isaac --episodes $E1 --grounding oracle --controller cartesian --out $R/runs/demo_a
$R/run.sh -m vrg.franka.run_isaac --episodes $E1,114:051_large_clamp,101:009_gelatin_box --grounding oracle --controller cartesian --refine --out $R/runs/demo_b
$R/run.sh -m vrg.franka.run_isaac --episodes $E1 --grounding oracle --controller native --refine --out $R/runs/demo_e
$R/run.sh -m vrg.franka.run_isaac --episodes 101:009_gelatin_box --grounding file:$G/qwen3vl-2b_point.json --controller cartesian --refine --out $R/runs/demo_vlm2b
$R/run.sh -m vrg.franka.run_isaac --episodes 101:009_gelatin_box --grounding file:$G/qwen3vl-32b_marks.json --controller cartesian --refine --out $R/runs/demo_vlm32b
echo "DEMO DONE"
