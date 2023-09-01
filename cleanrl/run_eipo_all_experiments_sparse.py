import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
# parser.add_argument("--bonus_type", default="icm", choices=["icm", "dynamics"])
parser.add_argument("--gpu", default=1)
args = parser.parse_args()

lams = [1.0]
seeds = [4, 3, 2, 1, 0]
envs = [
    "SparseHopper-v2",
    "SparseWalker2d-v2",
    "SparseHalfCheetah-v2",
    "SparseAnt-v2",
    "SparseHumanoid-v2",
]
bonus_types = [
    'icm',
    'dynamics',
    'disagreement'
]

# subprocess.run("conda activate mujoco", check=True, shell=True)

for env in envs:
    for bonus_type in bonus_types:
        for seed in seeds:
            for lam in lams:
                command = [
                    f"CUDA_VISIBLE_DEVICES={args.gpu}",
                    f"python eipo_rnd_mujoco.py",
                    f"--env-id {env}",
                    f"--bonus_factor {lam}",
                    f"--seed {seed}",
                    f"--bonus_type {bonus_type}",
                    # f"--use-ppo-hyper"
                ]

                # print(" ".join(command))
                print(" ".join(command))
                subprocess.run(" ".join(command), check=True, shell=True)