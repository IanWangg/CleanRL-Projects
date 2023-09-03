import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
# parser.add_argument("--bonus_type", default="icm", choices=["icm", "dynamics"])
parser.add_argument("--gpu", default=1)
parser.add_argument("--env", default=None, type=str)
parser.add_argument("--use-ppo-hyper", action='store_true')
parser.add_argument("--seed", default=None, type=int, nargs="*")
parser.add_argument("--bonus_type", default=None)
args = parser.parse_args()

lams = [1.0]
seeds = [4, 3, 2, 1, 0] if args.seed is None else args.seed
envs = [
    "Hopper-v2",
    "Walker2d-v2",
    "HalfCheetah-v2",
    "Ant-v2",
    "Humanoid-v2",
] if args.env is None else [args.env]

bonus_types = [
    'icm',
    'dynamics',
    'disagreement'
] if args.bonus_type is None else [args.bonus_type]

print(seeds, envs, bonus_types)
# raise ValueError

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
                    "--use-ppo-hyper" if args.use_ppo_hyper else "",
                ]

                # print(" ".join(command))
                print(" ".join(command))
                subprocess.run(" ".join(command), check=True, shell=True)