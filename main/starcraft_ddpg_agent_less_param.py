import numpy as np

from rl.agents.multiAgents.multi_ddpg import MA_DDPGAgent
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint

import keras
from keras.optimizers import Adam

from env.env_starcarft import StarCraftEnvironment
from main import GlobalVariable as gvar

from rl.MultiAgentPolicy import starcraft_multiagent_eGreedyPolicy
from rl.policy import LinearAnnealedPolicy
from env.ac_networks import actor_net, critic_net
from rl.callbacks import TrainHistoryLogCallback

import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = gvar.cuda_device

# define environment
agent_name = 'starcraft_minigame_vulture_zealot'
env_details = {'ally': ['verture']*2, 'enemy': ['zealot']*3, 'state_dim': (64, 64, 3 + 2),
               'dim_state_2D': (64, 64, 5), 'dim_state_1D': (2, )}
env = StarCraftEnvironment(agent_name, env_details)

# define policy
policy_minigame = starcraft_multiagent_eGreedyPolicy(nb_agents=env.nb_agents,
                                                     nb_actions=env.nb_actions)
policy = LinearAnnealedPolicy(policy_minigame, attr='eps', value_max=1.,
                              value_min=.1, value_test=.05, nb_steps=400000)
test_policy = starcraft_multiagent_eGreedyPolicy(nb_agents=env.nb_agents,
                                                 nb_actions=env.nb_actions,
                                                 eps=0.05)

def rl_learn(a=None):
    keras.backend.clear_session()
    actor = actor_net(env)
    critic = critic_net(env)

    # build actor/critic network
    model_path = 'save_model/{}_weights.h5f'.format(agent_name)
    memory = SequentialMemory(limit=50000, window_length=1)
    histcallback = TrainHistoryLogCallback(file_path='save_model/', plot_interval=1000)
    chkpoint = ModelIntervalCheckpoint(filepath=model_path, interval=10000)

    # define policy
    policy_minigame = starcraft_multiagent_eGreedyPolicy(nb_agents=env.nb_agents,
                                                         nb_actions=env.nb_actions)
    policy = LinearAnnealedPolicy(policy_minigame, attr='eps', value_max=1.,
                                  value_min=.1, value_test=.05, nb_steps=100000)
    test_policy = starcraft_multiagent_eGreedyPolicy(nb_agents=env.nb_agents,
                                                     nb_actions=env.nb_actions,
                                                     eps=0.05)

    agent = MA_DDPGAgent(nb_agents=env.nb_agents, nb_actions=env.nb_actions,
                         actor=actor, critic=critic, action_type='discrete',
                         critic_action_input=critic.inputs[2:4], train_interval=10,
                         batch_size=128, memory=memory, nb_steps_warmup_critic=5000, reward_factor=0.1,
                         nb_steps_warmup_actor=5000, policy=policy, test_policy=test_policy,
                         gamma=.99, target_model_update=1e-3)

    agent.compile([Adam(lr=5e-5), Adam(lr=5e-5)], metrics=['mae'])

    actor.summary()
    critic.summary()

    time.sleep(1)
    hist_train = agent.fit(env, nb_steps=6000000, nb_max_episode_steps=2000,
                           visualize=False, verbose=2, callbacks=[histcallback, chkpoint])

    np.save('save_model/hist_train.npy', hist_train.history)
    # After training is done, we save the final weights.
    agent.save_weights(model_path, overwrite=True)
    '''
    # Finally, evaluate our algorithm for 5 episodes.
    hist_test = agent.test(env, nb_episodes=2000, verbose=1, visualize=False)
    np.save('save_model/hist_test.npy', hist_test.history)
    '''