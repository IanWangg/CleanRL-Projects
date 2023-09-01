# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from gymnasium.wrappers.normalize import RunningMeanStd

# for dealing with the pointmass observation
from gymnasium.wrappers import FilterObservation as filter_wrapper
from gymnasium.wrappers import StepAPICompatibility as step_wrapper

from maps import get_maps

import copy

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    # EIPO arguments
    parser.add_argument("--alpha-lr", type=float, default=0.01,
        help="learning rate of lagrangian multiplier")
    parser.add_argument("--alpha-g-clip", type=float, default=0.05,
        help="clip on alpha update")
    parser.add_argument("--alpha-clip", type=float, default=10,
        help="clip on alpha value")

    # RND arguments
    parser.add_argument("--update-proportion", type=float, default=0.25,
        help="proportion of exp used for predictor update")
    parser.add_argument("--int-coef", type=float, default=1.0,
        help="coefficient of extrinsic reward")
    parser.add_argument("--ext-coef", type=float, default=2.0,
        help="coefficient of intrinsic reward")
    parser.add_argument("--int-gamma", type=float, default=0.99,
        help="Intrinsic reward discount rate")
    parser.add_argument("--num-iterations-obs-norm-init", type=int, default=50,
        help="number of iterations to initialize the observations normalization parameters")

    parser.add_argument("--bonus_type", default="disagreement", choices=["icm", "dynamics", "disagreement"])
    parser.add_argument("--bonus_factor", default=1.0, type=float)

    parser.add_argument("--use-ppo-hyper", action="store_true")

    args = parser.parse_args()

    if not args.use_ppo_hyper:
        args.update_epochs = 4
        args.num_envs = 128
        args.num_steps = 128
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = 32
        args.gamma = 0.999
        args.int_gamma = 0.99
    else:
        args.batch_size = int(args.num_envs * args.num_steps)
        args.gamma = 0.999
        args.int_gamma = 0.99
        args.minibatch_size = 64
        args.update_epochs = 10
        args.num_envs = 32
        args.num_steps = 32
    # fmt: on
    return args

def make_env(env_id, idx, capture_video, run_name, gamma): 
    def thunk():
        if capture_video:
            env = gym.make('PointMaze_UMaze-v3', maze_map=get_maps(env_id), render_mode="rgb_array", max_episode_steps=1000, continuing_task=False)
        else:
            env = gym.make('PointMaze_UMaze-v3', maze_map=get_maps(env_id))
        
        env = gym.wrappers.FilterObservation(env, ['observation'])
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, init_alpha=0.5):
        super().__init__()
        self.network = nn.Sequential(layer_init(nn.Linear(envs.single_observation_space.shape[0], 64), std=0.1), nn.ReLU())
        # self.network = nn.Sequential(layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)), nn.ReLU())
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(64, 64), std=0.1), nn.ReLU())
        self.actor_ei = nn.Sequential(
            layer_init(nn.Linear(64, 64), std=0.01),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.shape[0]), std=0.01),
        )
        self.actor_e = nn.Sequential(
            layer_init(nn.Linear(64, 64), std=0.01),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.shape[0]), std=0.01),
        )
        self.critic_ei_ext = layer_init(nn.Linear(64, 1), std=0.01)
        self.critic_ei_int = layer_init(nn.Linear(64, 1), std=0.01)
        self.critic_e_ext = layer_init(nn.Linear(64, 1), std=0.01)

        self.actor_e_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.actor_ei_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        
        self.rollout_by_pi_e = True
        self.alpha = init_alpha

    def get_action_and_value(self, x, action=None):
        # print(x.shape)
        hidden = self.network(x)
        features = self.extra_layer(hidden)

        # EI
        mean_ei = self.actor_ei(hidden)
        std_ei = torch.exp(self.actor_ei_logstd)
        probs_ei = Normal(mean_ei, std_ei)

        # E
        mean_e = self.actor_e(hidden)
        std_e = torch.exp(self.actor_e_logstd)
        probs_e = Normal(mean_e, std_e)
        
        if action is None:
          action_ei = probs_ei.sample()
          action_e = probs_e.sample()
        else:
          action_ei = action
          action_e = action
            
        return (
            action_ei,
            action_e,
            probs_ei.log_prob(action_ei).sum(1),
            probs_e.log_prob(action_e).sum(1),
            probs_ei.entropy(),
            probs_e.entropy(),
            self.critic_ei_ext(features + hidden),
            self.critic_ei_int(features + hidden),
            self.critic_e_ext(features + hidden)
        )

    def get_value(self, x):
        # print(x.shape, args.num_envs, envs.single_action_space.shape[0], np.array(envs.single_observation_space.shape).prod())
        hidden = self.network(x)
        features = self.extra_layer(hidden)
        return self.critic_ei_ext(features + hidden), self.critic_ei_int(features + hidden), self.critic_e_ext(features + hidden)

    def is_rollout_by_pi_e(self):
        return self.rollout_by_pi_e

    def maybe_switch_rollout_pi(self, 
        old_max_objective_value, new_max_objective_value,
        old_min_objective_value, new_min_objective_value):
        if self.is_rollout_by_pi_e() and old_max_objective_value is not None:
          if (new_max_objective_value - old_max_objective_value) < 0:
            self.rollout_by_pi_e = False
            print("Switch to pi_EI")
        elif not self.is_rollout_by_pi_e() and old_min_objective_value is not None:
          if (new_min_objective_value - old_min_objective_value) < 0:
            self.rollout_by_pi_e = True
            print("Switch to pi_E")


class DisagreementModel(nn.Module):
    def __init__(self, envs, n_models=5):
        super().__init__()
        state_dim = envs.single_observation_space.shape[0]
        action_dim = envs.single_action_space.shape[0]
        self.ensemble = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim + action_dim, 256),
                          nn.ReLU(), nn.Linear(256, state_dim))
            for _ in range(n_models)
        ])

    def forward(self, obs, action, next_obs):
        #import ipdb; ipdb.set_trace()
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        errors = 0
        preds = []
        for model in self.ensemble:
            next_obs_hat = model(torch.cat([obs, action], dim=-1))
            preds.append(next_obs_hat)
            model_error = torch.norm(next_obs_hat - next_obs, dim=-1, p=2, keepdim=True)
            errors += model_error

        preds = torch.stack(preds, dim=0)

        return errors, torch.var(preds, dim=0).mean(dim=-1)
    
class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.bonus_type}__{args.bonus_factor}__{args.seed}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = envpool.make(
    #     args.env_id,
    #     env_type="gym",
    #     num_envs=args.num_envs,
    #     # episodic_life=True,
    #     # reward_clip=True,
    #     seed=args.seed,
    #     # repeat_action_probability=0.25,
    # )
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, capture_video=False, run_name="default", gamma=1.0) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    envs.num_envs = args.num_envs
    # envs.single_action_space = envs.action_space
    # envs.single_observation_space = envs.observation_space
    # envs = RecordEpisodeStatistics(envs)
    # assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    if args.bonus_type in ["icm", "dynamics"]:
        icm_model = ICMModel(envs).to(device)
    else:
        icm_model = DisagreementModel(envs).to(device)
    combined_parameters = list(agent.parameters()) + list(icm_model.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, envs.single_observation_space.shape[0]))
    action_rms = RunningMeanStd(shape=(1, envs.single_action_space.shape[0]))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_ei = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    actions_e = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs_ei = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_e = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ei_ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ei_int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    e_ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # need to store all next_obs
    next_obs_place_holder = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # next_obs, _ = envs.reset()
    next_obs = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    '''if not os.path.exists(f".atari_norm/{args.env_id}.pkl"):
      print("Start to initialize observation normalization parameter.....")
      next_ob = []
      for step in tqdm(range(args.num_steps * args.num_iterations_obs_norm_init)):
          acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
          s, r, d, _ = envs.step(acs)
          next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()

          if len(next_ob) % (args.num_steps * args.num_envs) == 0:
              next_ob = np.stack(next_ob)
              obs_rms.update(next_ob)
              next_ob = []
      os.makedirs(".atari_norm", exist_ok=True)
      with open(f".atari_norm/{args.env_id}.pkl", "wb") as f:
        pickle.dump(obs_rms, f)
    else:
      print(f"Load obs_rms from .atari_norm/{args.env_id}.pkl")      
      with open(f".atari_norm/{args.env_id}.pkl", "rb") as f:
        obs_rms = pickle.load(f)
    print("End to initialize...")'''
    old_max_objective_value = None
    new_max_objective_value = None
    old_min_objective_value = None
    new_min_objective_value = None

    # main training loop
    performance = []
    buffer = []
    np_filename = f"eipo_pointmass_{args.bonus_type}_{args.env_id}_{args.seed}{'_ppo_hyper' if args.use_ppo_hyper else ''}"

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # print(step, obs.shape, obs[step].shape, envs.single_action_space)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # print(next_obs.shape, obs[step].shape)
                value_ei_ext, value_ei_int, value_e_ext = agent.get_value(obs[step])
                ei_ext_values[step], ei_int_values[step], e_ext_values[step] = (
                    value_ei_ext.flatten(),
                    value_ei_int.flatten(),
                    value_e_ext.flatten(),
                )
                action_ei, action_e, logprob_ei, logprob_e, _, _, _, _, _ = agent.get_action_and_value(obs[step])

            actions_ei[step] = action_ei
            actions_e[step] = action_e
            # print(logprob_ei.shape, logprobs_ei.shape)
            logprobs_ei[step] = logprob_ei
            logprobs_e[step] = logprob_e

            # TRY NOT TO MODIFY: execute the game and log data.
            rollout_action = action_e if agent.is_rollout_by_pi_e() else action_ei 

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # store the next observation
            next_obs_place_holder[step] = next_obs

            # deal with the intrinsic bonus
            icm_obs = (
                (
                    (obs[step] - torch.from_numpy(obs_rms.mean).to(device))
                    / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                )
            ).float()
            icm_next_obs = (
                (
                    (next_obs - torch.from_numpy(obs_rms.mean).to(device))
                    / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                )
            ).float()
            icm_action = (
                (
                    (rollout_action - torch.from_numpy(action_rms.mean).to(device))
                    / torch.sqrt(torch.from_numpy(action_rms.var).to(device))
                )
            ).float()

            if args.bonus_type in ["icm", "dynamics"]:
                forward_error, _ = icm_model(
                    # obs[step], 
                    # rollout_action, 
                    # next_obs
                    icm_obs,
                    icm_action,
                    icm_next_obs,
                )
            else:
                # for disagreement model, we use the estimation variance as bonus
                _, forward_error = icm_model(
                    # obs[step], 
                    # rollout_action, 
                    # next_obs
                    icm_obs,
                    icm_action,
                    icm_next_obs,
                )
            
            forward_error = forward_error * args.bonus_factor # scale the bonus

            curiosity_rewards[step] = forward_error.view(-1)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                # take the negative episode length as return, the shorter the better
                performance.append([global_step, -item["episode"]["l"]])
                np.save(f"./eipo_pointmass_results/{np_filename}", performance)
                buffer.append(copy.deepcopy(obs))
                np.save(f"./eipo_buffer/{np_filename}_online_buffer", buffer)

        curiosity_reward_per_env = np.array(
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
        )
        mean, std, count = (
            np.mean(curiosity_reward_per_env),
            np.std(curiosity_reward_per_env),
            len(curiosity_reward_per_env),
        )
        reward_rms.update_from_moments(mean, std**2, count)

        curiosity_rewards /= np.sqrt(reward_rms.var)

        # bootstrap value if not done
        with torch.no_grad():
            next_value_ei_ext, next_value_ei_int, next_value_e_ext = agent.get_value(next_obs)
            next_value_ei_ext, next_value_ei_int, next_value_e_ext = next_value_ei_ext.reshape(1, -1), next_value_ei_int.reshape(1, -1), next_value_e_ext.reshape(1, -1)
            ei_ext_advantages = torch.zeros_like(rewards, device=device)
            ei_int_advantages = torch.zeros_like(curiosity_rewards, device=device)
            ei_advantages = torch.zeros_like(rewards, device=device) # U_{max}
            ei_ext_lastgaelam = 0
            ei_int_lastgaelam = 0
            e_ext_advantages = torch.zeros_like(rewards, device=device)
            e_advantages = torch.zeros_like(rewards, device=device) # U_min       
            e_ext_lastgaelam = 0            
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ei_ext_nextnonterminal = 1.0 - next_done
                    ei_int_nextnonterminal = 1.0
                    ei_ext_nextvalues = next_value_ei_ext
                    ei_int_nextvalues = next_value_ei_int
                    e_ext_nextnonterminal = 1.0 - next_done                    
                    e_ext_nextvalues = next_value_e_ext
                else:
                    ei_ext_nextnonterminal = 1.0 - dones[t + 1]
                    ei_int_nextnonterminal = 1.0
                    ei_ext_nextvalues = ei_ext_values[t + 1]
                    ei_int_nextvalues = ei_int_values[t + 1]
                    e_ext_nextnonterminal = 1.0 - dones[t + 1]
                    e_ext_nextvalues = e_ext_values[t + 1]
                ei_ext_delta = rewards[t] + args.gamma * ei_ext_nextvalues * ei_ext_nextnonterminal - ei_ext_values[t]
                ei_int_delta = curiosity_rewards[t] + args.int_gamma * ei_int_nextvalues * ei_int_nextnonterminal - ei_int_values[t]
                e_ext_delta = rewards[t] + args.gamma * e_ext_nextvalues * e_ext_nextnonterminal - e_ext_values[t]
                ei_ext_advantages[t] = ei_ext_lastgaelam = (
                    ei_ext_delta + args.gamma * args.gae_lambda * ei_ext_nextnonterminal * ei_ext_lastgaelam
                )
                ei_int_advantages[t] = ei_int_lastgaelam = (
                    ei_int_delta + args.int_gamma * args.gae_lambda * ei_int_nextnonterminal * ei_int_lastgaelam
                )
            
            # U_max = (1 + alpha) * r_E + r_I + \gamma * \alpha * V^pi_E_E(s') - \alpha * V^\pi_E(s)
            #   = alpha * (r_E + \gamma * V^pi_E_E(s') - V^\pi_E_E(s)) + r_E + r_I
            #   = alpha * A^pi_E(s) + r_E + r_I
            ei_advantages = (rewards + curiosity_rewards) + agent.alpha * e_ext_advantages

            # U_min = (\alpha * rE + gamma * ((1+\alpha) * V_E + V_I)(s') - ((1+\alpha) * V_E + V_I)(s))
            #   = \alpha * (rE + gamma * V_E(s') - V_E(s)) + gamma * (V_E + V_I)(s') - (V_E + V_I)(s)
            #   = \alpha * A_E(s) + A_{E+I}(s) - (r_{E} + r_I)
            #   = (1 + \alpha) * A_E(s) + A_I(s) - (r_E + r_I)
            e_advantages = (1 + agent.alpha) * ei_ext_advantages + \
                ei_int_advantages - \
                (rewards + curiosity_rewards)


            ei_ext_returns = ei_ext_advantages + ei_ext_values
            ei_int_returns = ei_int_advantages + ei_int_values
            e_ext_returns = e_ext_advantages + e_ext_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs_ei = logprobs_ei.reshape(-1)
        b_actions_ei = actions_ei.reshape((-1,) + envs.single_action_space.shape)
        b_ei_ext_advantages = ei_ext_advantages.reshape(-1)
        b_ei_int_advantages = ei_int_advantages.reshape(-1)
        b_ei_advantages = ei_advantages.reshape(-1)
        b_ei_ext_returns = ei_ext_returns.reshape(-1)
        b_ei_int_returns = ei_int_returns.reshape(-1)
        b_ei_ext_values = ei_ext_values.reshape(-1)

        b_logprobs_e = logprobs_e.reshape(-1)
        b_actions_e = actions_e.reshape((-1,) + envs.single_action_space.shape)
        b_e_ext_advantages = e_ext_advantages.reshape(-1)
        b_e_advantages = e_advantages.reshape(-1)
        b_e_ext_returns = e_ext_returns.reshape(-1)
        b_e_ext_values = e_ext_values.reshape(-1)

        b_ei_eipo_advantages = b_ei_advantages
        b_ei_ppo_advantages = b_ei_int_advantages * args.int_coef + b_ei_ext_advantages * args.ext_coef

        b_e_eipo_advantages = b_e_advantages
        b_e_ppo_advantages = b_e_ext_advantages

        obs_rms.update(b_obs[:, :].cpu().numpy())
        action_rms.update(b_actions_e[:, :].cpu().numpy())
        action_rms.update(b_actions_ei[:, :].cpu().numpy())

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        b_next_obs = next_obs_place_holder.reshape((-1,) + envs.single_observation_space.shape)

        clipfracs_ei_ei = []
        clipfracs_e_e = []
        clipfracs_ei_e = []
        clipfracs_e_ei = []
        alpha_derivatives = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # import pdb; pdb.set_trace()
                # Depends on who collect the data
                b_actions = b_actions_e if agent.is_rollout_by_pi_e() else b_actions_ei 
                # predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                # print(b_obs[mb_inds].shape, b_actions[mb_inds].shape, b_next_obs[mb_inds].shape)
                icm_obs = (
                    (
                        (b_obs[mb_inds] - torch.from_numpy(obs_rms.mean).to(device))
                        / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                    )
                ).float()
                icm_next_obs = (
                    (
                        (b_next_obs[mb_inds] - torch.from_numpy(obs_rms.mean).to(device))
                        / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                    )
                ).float()
                icm_action = (
                    (
                        (b_actions[mb_inds] - torch.from_numpy(action_rms.mean).to(device))
                        / torch.sqrt(torch.from_numpy(action_rms.var).to(device))
                    )
                ).float()

                if args.bonus_type in ["icm", "dynamics"]:
                    forward_error, inverse_error = icm_model(icm_obs, icm_action, icm_next_obs)
                    forward_loss = forward_error + inverse_error
                else:
                    error, _ = icm_model(icm_obs, icm_action, icm_next_obs)
                    forward_loss = error

                mask = torch.rand(len(forward_loss), device=device)
                mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                ) 

                _, _, newlogprob_ei, newlogprob_e, entropy_ei, entropy_e, new_ei_ext_values, new_ei_int_values, new_e_ext_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                # EI_new / EI_old
                logratio_ei_ei = newlogprob_ei - b_logprobs_ei[mb_inds]
                ratio_ei_ei = logratio_ei_ei.exp()
                # EI_new / E_old
                logratio_ei_e = newlogprob_ei - b_logprobs_e[mb_inds]
                ratio_ei_e = logratio_ei_e.exp()
                # E_new / E_old
                logratio_e_e = newlogprob_e - b_logprobs_e[mb_inds]
                ratio_e_e = logratio_e_e.exp()
                # E_new / EI_old
                logratio_e_ei = newlogprob_e - b_logprobs_ei[mb_inds]
                ratio_e_ei = logratio_e_ei.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl_ei_ei = (-logratio_ei_ei).mean()
                    approx_kl_ei_ei = ((ratio_ei_ei - 1) - logratio_ei_ei).mean()
                    clipfracs_ei_ei += [((ratio_ei_ei - 1.0).abs() > args.clip_coef).float().mean().item()]
                    
                    old_approx_kl_ei_e = (-logratio_ei_e).mean()
                    approx_kl_ei_e = ((ratio_ei_e - 1) - logratio_ei_e).mean()
                    clipfracs_ei_e += [((ratio_ei_e - 1.0).abs() > args.clip_coef).float().mean().item()]
                    
                    old_approx_kl_e_e = (-logratio_e_e).mean()
                    approx_kl_e_e = ((ratio_e_e - 1) - logratio_e_e).mean()
                    clipfracs_e_e += [((ratio_e_e - 1.0).abs() > args.clip_coef).float().mean().item()]
                    
                    old_approx_kl_e_ei = (-logratio_e_ei).mean()
                    approx_kl_e_ei = ((ratio_e_ei - 1) - logratio_e_ei).mean()
                    clipfracs_e_ei += [((ratio_e_ei - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_ei_eipo_advantages = b_ei_eipo_advantages[mb_inds]
                mb_ei_ppo_advantages = b_ei_ppo_advantages[mb_inds]
                mb_e_eipo_advantages = b_e_eipo_advantages[mb_inds]
                mb_e_ppo_advantages = b_e_ppo_advantages[mb_inds]
                if args.norm_adv:
                    mb_ei_eipo_advantages = (mb_ei_eipo_advantages - mb_ei_eipo_advantages.mean()) / (mb_ei_eipo_advantages.std() + 1e-8)
                    mb_ei_ppo_advantages = (mb_ei_ppo_advantages - mb_ei_ppo_advantages.mean()) / (mb_ei_ppo_advantages.std() + 1e-8)
                    mb_e_eipo_advantages = (mb_e_eipo_advantages - mb_e_eipo_advantages.mean()) / (mb_e_eipo_advantages.std() + 1e-8)
                    mb_e_ppo_advantages = (mb_e_ppo_advantages - mb_e_ppo_advantages.mean()) / (mb_e_ppo_advantages.std() + 1e-8)

                # Policy loss
                if agent.is_rollout_by_pi_e(): # Max stage
                  pg_loss1_ei_eipo = -mb_ei_eipo_advantages * ratio_ei_e
                  pg_loss2_ei_eipo = -mb_ei_eipo_advantages * torch.clamp(ratio_ei_e, 1 - args.clip_coef, 1 + args.clip_coef)
                  pg_loss_ei_eipo = torch.max(pg_loss1_ei_eipo, pg_loss2_ei_eipo).mean()
                  
                  pg_loss1_e_ppo = -mb_e_ppo_advantages * ratio_e_e
                  pg_loss2_e_ppo = -mb_e_ppo_advantages * torch.clamp(ratio_e_e, 1 - args.clip_coef, 1 + args.clip_coef)
                  pg_loss_e_ppo = torch.max(pg_loss1_e_ppo, pg_loss2_e_ppo).mean()
                  pg_loss = pg_loss_ei_eipo + pg_loss_e_ppo
                  alpha_derivative = mb_e_ppo_advantages.mean().detach().cpu().item()
                  alpha_derivatives.append(alpha_derivative)
                  # For logging
                  pg_ei_loss = pg_loss_ei_eipo
                  pg_e_loss = pg_loss_e_ppo
                else: # Min stage
                  pg_loss1_e_eipo = -mb_e_eipo_advantages * ratio_e_ei
                  pg_loss2_e_eipo = -mb_e_eipo_advantages * torch.clamp(ratio_e_ei, 1 - args.clip_coef, 1 + args.clip_coef)
                  pg_loss_e_eipo = torch.max(pg_loss1_e_eipo, pg_loss2_e_eipo).mean()
                  
                  pg_loss1_ei_ppo = -mb_ei_ppo_advantages * ratio_ei_ei
                  pg_loss2_ei_ppo = -mb_ei_ppo_advantages * torch.clamp(ratio_ei_ei, 1 - args.clip_coef, 1 + args.clip_coef)
                  pg_loss_ei_ppo = torch.max(pg_loss1_ei_ppo, pg_loss2_ei_ppo).mean()
                  pg_loss = pg_loss_e_eipo + pg_loss_ei_ppo
                  # For logging
                  pg_ei_loss = pg_loss_ei_ppo
                  pg_e_loss = pg_loss_e_eipo

                # Value loss
                new_ei_ext_values, new_ei_int_values = new_ei_ext_values.view(-1), new_ei_int_values.view(-1)
                new_e_ext_values = new_e_ext_values.view(-1)
                if args.clip_vloss:
                    ei_ext_v_loss_unclipped = (new_ei_ext_values - b_ei_ext_returns[mb_inds]) ** 2
                    ei_ext_v_clipped = b_ei_ext_values[mb_inds] + torch.clamp(
                        new_ei_ext_values - b_ei_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ei_ext_v_loss_clipped = (ei_ext_v_clipped - b_ei_ext_returns[mb_inds]) ** 2
                    ei_ext_v_loss_max = torch.max(ei_ext_v_loss_unclipped, ei_ext_v_loss_clipped)
                    ei_ext_v_loss = 0.5 * ei_ext_v_loss_max.mean()

                    e_ext_v_loss_unclipped = (new_e_ext_values - b_e_ext_returns[mb_inds]) ** 2
                    e_ext_v_clipped = b_e_ext_values[mb_inds] + torch.clamp(
                        new_e_ext_values - b_e_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    e_ext_v_loss_clipped = (e_ext_v_clipped - b_e_ext_returns[mb_inds]) ** 2
                    e_ext_v_loss_max = torch.max(e_ext_v_loss_unclipped, e_ext_v_loss_clipped)
                    e_ext_v_loss = 0.5 * e_ext_v_loss_max.mean()
                else:
                    ei_ext_v_loss = 0.5 * ((new_ei_ext_values - b_ei_ext_returns[mb_inds]) ** 2).mean()
                    e_ext_v_loss = 0.5 * ((new_e_ext_values - b_e_ext_returns[mb_inds]) ** 2).mean()

                ei_int_v_loss = 0.5 * ((new_ei_int_values - b_ei_int_returns[mb_inds]) ** 2).mean()
                ei_v_loss = ei_ext_v_loss + ei_int_v_loss
                e_v_loss = e_ext_v_loss
                v_loss = ei_v_loss + e_v_loss

                entropy_ei = entropy_ei.mean()
                entropy_e = entropy_e.mean()
                entropy_loss = entropy_ei + entropy_e

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()

            if args.target_kl is not None:
                if agent.is_rollout_by_pi_e():
                  if approx_kl_e_e > args.target_kl or approx_kl_ei_e > args.target_kl:
                    break
                else:
                  if approx_kl_ei_ei > args.target_kl or approx_kl_e_ei > args.target_kl:
                    break
         
        # Check if we need to switch the policy            
        old_is_rollout_by_pi_e = agent.is_rollout_by_pi_e() # True: max stage, False: min stage
        agent.maybe_switch_rollout_pi(old_max_objective_value, ei_advantages.mean(),
          old_min_objective_value, e_advantages.mean())      
        new_is_rollout_by_pi_e = agent.is_rollout_by_pi_e() # True: max stage, False: min stage
        
        # Update alpha (only after max stage)
        if old_is_rollout_by_pi_e and (not new_is_rollout_by_pi_e):
          agent.alpha = agent.alpha - args.alpha_lr * np.clip(np.mean(alpha_derivatives), -args.alpha_g_clip, args.alpha_g_clip)
          agent.alpha = np.clip(agent.alpha, -args.alpha_clip, args.alpha_clip)
              
        old_max_objective_value = ei_advantages.mean().item()
        old_min_objective_value = e_advantages.mean().item()

        

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        writer.add_scalar("charts/alpha", agent.alpha, global_step)
        writer.add_scalar("charts/max_value_diff", (ei_advantages.mean().item() - old_max_objective_value), global_step)
        writer.add_scalar("charts/min_value_diff", (e_advantages.mean().item() - old_min_objective_value), global_step)
        writer.add_scalar("charts/rollout_by_pi_e", old_is_rollout_by_pi_e, global_step)

        writer.add_scalar("losses/ei_value_loss", ei_v_loss.item(), global_step)
        writer.add_scalar("losses/ei_policy_loss", pg_ei_loss.item(), global_step)
        writer.add_scalar("losses/ei_entropy", entropy_ei.item(), global_step)

        writer.add_scalar("losses/e_value_loss", e_v_loss.item(), global_step)
        writer.add_scalar("losses/e_policy_loss", pg_e_loss.item(), global_step)
        writer.add_scalar("losses/e_entropy", entropy_e.item(), global_step)

        writer.add_scalar("losses/ei_ei_old_approx_kl", old_approx_kl_ei_ei.item(), global_step)
        writer.add_scalar("losses/ei_ei_approx_kl", approx_kl_ei_ei.item(), global_step)
        writer.add_scalar("losses/ei_e_old_approx_kl", old_approx_kl_ei_e.item(), global_step)
        writer.add_scalar("losses/ei_e_approx_kl", approx_kl_ei_e.item(), global_step)
        writer.add_scalar("losses/e_ei_old_approx_kl", old_approx_kl_ei_ei.item(), global_step)
        writer.add_scalar("losses/e_ei_approx_kl", approx_kl_ei_ei.item(), global_step)
        writer.add_scalar("losses/e_e_old_approx_kl", old_approx_kl_ei_e.item(), global_step)
        writer.add_scalar("losses/e_e_approx_kl", approx_kl_ei_e.item(), global_step)

        writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

       

    envs.close()
    writer.close()
