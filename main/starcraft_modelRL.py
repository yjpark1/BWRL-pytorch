from env.env_starcarft_pytorch import StarCraftEnvironment
from rl.networks.ac_network_model import ActorNetwork, CriticNetwork
from rl.agent.model_ddpg import Trainer
from main import GlobalVariable as gvar
import numpy as np
import torch
import time
from rl.replay_buffer import SequentialMemory
from rl import arglist
from rl.utils import OUNoise
from copy import deepcopy
torch.set_default_tensor_type('torch.cuda.FloatTensor')

ou_xy = OUNoise(action_dimension=2, theta=0.15, sigma=0.2)


def rl_learn(cnt=0):
    torch.cuda.empty_cache()
    # load scenario from script
    scenario_name = 'TV2vsPZ3'
    env_details = {'ally': ['verture']*2,
                   'enemy': ['zealot']*3,
                   'state_dim': (2, 36)}

    env = StarCraftEnvironment(agent_name=scenario_name, env_details=env_details)

    actor = ActorNetwork(nb_agents=env.nb_agents, input_dim=36, out_dim=[2, 3])
    critic = CriticNetwork(nb_agents=env.nb_agents, input_dim=36 + 5, out_dim=1)
    memory = SequentialMemory(limit=1000000)
    agent = Trainer(actor, critic, memory, noise=ou_xy)

    # initialize history
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.nb_agents)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    terminal_reward = []
    episode_loss = []

    while True:
        if gvar.service_flag == 0:
            time.sleep(1e-2)
        else:
            obs = env.reset()
            break

    episode_step = 0
    train_step = 0
    nb_episode = 0

    verbose_step = False
    verbose_episode = True
    t_start = time.time()

    # log = open('results/train_log.txt', 'w')
    # log.write('train start... \n')
    # log.close()

    print('Starting iterations...')
    while True:
        # get action
        obs = agent.process_obs(obs)
        actions = agent.get_exploration_action(obs)

        # environment step
        while True:
            if gvar.service_flag == 0:
                time.sleep(1e-2)
            else:
                new_obs, rewards, done, info = env.step(actions)
                break

        actions = agent.process_action(actions)
        rewards = agent.process_reward(rewards)
        rewards = rewards.mean()
        episode_step += 1

        terminal = arglist.max_episode_len and episode_step >= arglist.max_episode_len
        terminal = agent.process_done(done or terminal)
        # collect experience
        # obs, actions, rewards, done
        agent.memory.append(obs, actions, rewards, terminal, training=True)
        # next observation
        obs = deepcopy(new_obs)

        # episode_rewards.append(rewards)
        rewards = rewards.item()
        for i, rew in enumerate([rewards]):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        # for save & print history
        terminal_verbose = terminal or done
        if terminal or done:
            terminal_reward.append(np.mean(rewards))
            # save terminal state
            # process observation
            obs = agent.process_obs(obs)
            # get action & process action
            actions = agent.get_exploration_action(obs)
            actions = agent.process_action(actions)
            # process rewards
            rewards = agent.process_reward(0.)
            rewards = rewards.mean().item()
            # process terminal
            agent.memory.append(obs, actions, rewards, agent.process_done(False), training=True)

            # reset environment
            while True:
                if gvar.service_flag == 0:
                    time.sleep(1e-2)
                else:
                    action_token = env._make_action_token(env.dummy_action)
                    gvar.release_action = True
                    gvar.action = action_token
                    obs = env.reset()
                    if sum(env.token_unit[:, 1] > 0) == len(env_details['enemy']) + len(env_details['ally']):
                        break

            episode_step = 0
            nb_episode += 1
            episode_rewards.append(0)

        # increment global step counter
        train_step += 1

        # update all trainers, if not in display or benchmark mode
        loss = [np.nan, np.nan]
        if (train_step > arglist.warmup_steps) and (train_step % 200 == 0):
            # optimize actor-critic
            loss = agent.optimize()
            loss = np.array([x.data.item() for x in loss])
            episode_loss.append(loss)

        if verbose_step:
            if loss == [np.nan, np.nan]:
                loss = ['--', '--']
            print('step: {}, actor_loss: {}, critic_loss: {}'.format(train_step, loss[0], loss[1]))

        elif verbose_episode:
            if terminal_verbose and (len(episode_rewards) % arglist.save_rate == 0):
                monitor_loss = np.mean(np.array(episode_loss)[-1000:], axis=0)

                msg1 = "steps: {}, episodes: {}, mean episode reward: {}, reward: {}, time: {}".format(
                    train_step, len(episode_rewards), round(np.mean(episode_rewards[-arglist.save_rate:]), 3),
                    round(np.mean(terminal_reward), 3), round(time.time() - t_start, 3))

                msg2 = "TD error: {}, c_model: {}, actorQ: {}, a_model: {}".format(
                    round(monitor_loss[2], 3),
                    round(monitor_loss[3], 3),
                    round(monitor_loss[4], 3),
                    round(monitor_loss[5], 3))
                msg = msg1 + ', ' + msg2
                print(msg)

                # save log
                # log = open('results/train_log.txt', 'a')
                # log.write(msg + '\n')
                # log.close()

                terminal_reward = []
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if nb_episode > arglist.num_episodes:
            np.save('results/iter_{}_episode_rewards.npy'.format(cnt), episode_rewards)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break


if __name__ == '__main__':
    for cnt in range(10):
        rl_learn(cnt)
