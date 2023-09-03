import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
# parser.add_argument("--bonus_type", default="icm", choices=["icm", "dynamics"])
parser.add_argument("--gpu", default=1)
parser.add_argument("--env", default=None, type=str)
parser.add_argument("--seeds", default=None, nargs="*")
args = parser.parse_args()

lams = [1.0]
seeds = [4, 3, 2, 1, 0] if args.seeds is None else args.seeds


if args.env is not None:
    envs = [args.env]

bonus_types = [
    'disagreement'
]

# subprocess.run("conda activate mujoco", check=True, shell=True)

for env in envs:
    for bonus_type in bonus_types:
        for seed in seeds:
            for lam in lams:
                command = [
                    f"CUDA_VISIBLE_DEVICES={args.gpu}",
                    f"python eipo_rnd_pointmass.py",
                    f"--env-id {env}",
                    f"--bonus_factor {lam}",
                    f"--seed {seed}",
                    f"--bonus_type {bonus_type}",
                ]

                # print(" ".join(command))
                print(" ".join(command))
                subprocess.run(" ".join(command), check=True, shell=True)