import argparse
import datetime

from tensorboardX import SummaryWriter

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from algo.ddpg import DDPG
from algo.sac import SAC
from common import *
from log_path import *
from env.chooseenv import make


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main(args):
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    env = make(args.game_name, conf=None)

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')
    ctrl_agent_index = [0, 1, 2]
    print(f'Agent control by the actor: {ctrl_agent_index}')
    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    print(f'Game board width: {width}')
    height = env.board_height
    print(f'Game board height: {height}')

    act_dim = env.get_action_dim()
    print(f'action dimension: {act_dim}')
    obs_dim = 26
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)

    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    writer = SummaryWriter(str(log_dir))
    save_config(args, log_dir)

    # model = DDPG(obs_dim, act_dim, ctrl_agent_num, args)
    model = SAC(obs_dim, act_dim, ctrl_agent_num, args)

    if args.load_model:
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
        model.load_model(load_dir, episode=args.load_model_run_episode)

    episode = 0
    return_logger = []
    best_avg_reward = 0
    while episode < args.max_episodes:

        # Receive initial observation state s1
        state = env.reset()

        # During training, since all agents are given the same obs, we take the state of 1st agent.
        # However, when evaluation in Jidi, each agent get its own state, like state[agent_index]: dict()
        # more details refer to https://github.com/jidiai/Competition_3v3snakes/blob/master/run_log.py#L68
        # state: list() ; state[0]: dict()
        state_to_training = state[0]

        # ======================= feature engineering =======================
        # since all snakes play independently, we choose first three snakes for training.
        # Then, the trained model can apply to other agents. ctrl_agent_index -> [0, 1, 2]
        # Noted, the index is different in obs. please refer to env description.
        obs = get_observations(state_to_training, ctrl_agent_index, obs_dim, height, width)

        episode += 1
        step = 0
        episode_reward = np.zeros(6)
        positive_reward = 0

        while True:

            # ================================== inference ========================================
            # For each agents i, select and execute action a:t,i = a:i,θ(s_t) + Nt
            logits = model.choose_action(obs)

            # ============================== add opponent actions =================================
            # we use rule-based greedy agent here. Or, you can switch to random agent.
            actions, control_actions = logits_greedy(state_to_training, logits, height, width)
            # actions = logits_random(act_dim, logits)

            # Receive reward [r_t,i]i=1~n and observe new state s_t+1
            next_state, reward, done, _, info = env.step(env.encode(actions))
            next_state_to_training = next_state[0]
            next_obs = get_observations(next_state_to_training, ctrl_agent_index, obs_dim, height, width)
            model.total_env_step += 1
            # ================================== reward shaping ========================================
            reward = np.array(reward)
            episode_reward += reward
            positive_reward += np.sum(reward > 0)

            step_reward = get_reward(info, ctrl_agent_index, reward, agent=model)

            done = np.array([done])

            # ================================== collect data ========================================
            # Store transition in R
            model.replay_buffer.push(obs, control_actions, step_reward, next_obs, done)
            
            if model.total_env_step - model.prev_env_step >= args.num_steps_between_train_calls and model.total_env_step >= model.batch_size:
                for _ in range(args.num_train_steps_per_train_call):
                    eval_statistics = model.update()
                model.prev_env_step = model.total_env_step
                for key in eval_statistics:
                    print(f"\t\t\t\t{key}: {eval_statistics[key]}")

            obs = next_obs
            state_to_training = next_state_to_training
            step += 1

            if args.episode_length <= step or (True in done):
                if len(return_logger) >= args.return_log_num:
                    return_logger.pop(0)
                    return_logger.append(np.sum(episode_reward[0:3]))
                else:
                    return_logger.append(np.sum(episode_reward[0:3]))
                pres_avg_reward = sum(return_logger) / len(return_logger)
                if pres_avg_reward > best_avg_reward:
                    best_avg_reward = pres_avg_reward

                print(f'[Episode {episode:05d}] total_reward: {sum(info["score"][0: 3]) - sum(info["score"][3: 6])} epsilon: {model.eps:.2f} \t step_reward: {step_reward}')
                print(f'\t\t\t\tsnake_1: {episode_reward[0]} '
                      f'snake_2: {episode_reward[1]} snake_3: {episode_reward[2]}')

                reward_tag = 'reward'
                writer.add_scalars(reward_tag, global_step=episode,
                                   tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                    'snake_3': episode_reward[2], 'total': np.sum(episode_reward[0:3])})

                if episode % args.save_interval == 0:
                    model.save_model(run_dir, episode)

                env.reset()
                model.length_adv = 0
                model.prev_opp_segments = None
                break
    with open('/home/lymao/Competition_3v3snakes-master/return_logger.txt', 'w') as f:
        f.write(f"alpha: {args.alpha} best_avg_reward: {best_avg_reward}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="ddpg", type=str, help="bicnet/ddpg")
    parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")

    parser.add_argument('--buffer_size', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epsilon', default=0.0, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)

    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    parser.add_argument("--load_model_run", default=2, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)

    # sac parameters
    parser.add_argument("--reward_scale", default=0.5, type=float)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--alpha_lr", default=0.0001, type=float)
    parser.add_argument("--return_log_num", default=10, type=int)
    parser.add_argument("--num_steps_between_train_calls", default=100, type=int)
    parser.add_argument("--num_train_steps_per_train_call", default=100, type=int)
    parser.add_argument("--normalize_factor", default=20.0, type=float)

    args = parser.parse_args()
    # # search best alpha
    # alpha_search_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0]
    # for alpha in alpha_search_list:
    #     args.alpha = alpha
    #     main(args)

    main(args)