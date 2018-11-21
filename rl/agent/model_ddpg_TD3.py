# reference: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/train.py
from __future__ import division
import torch
import numpy as np
import shutil
from rl import arglist
import copy
from rl.utils import to_categorical
from util.CustomLog import cLogger

logger = cLogger.getLogger()
GAMMA = 0.95
TAU = 0.001


class Trainer:
    def __init__(self, actor, critic, memory, noise):
        """
        DDPG for categorical action
        """
        self.device = torch.device('cuda:0')
        self.memory = memory

        self.iter = 0
        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), arglist.learning_rate)

        self.critic = critic.to(self.device)
        self.target_critic1 = copy.deepcopy(critic).to(self.device)
        self.target_critic2 = copy.deepcopy(critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), arglist.learning_rate)

        # set train flag
        self.actor.train()
        self.critic.train()

        self.target_actor.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()
        self.n_updates = 0
        self.eps = 1

        self.noise = noise

    def soft_update(self, target, source, tau):
        """
        Copies the parameters from source network (x) to target network (y) using the below update
        y = TAU*x + (1 - TAU)*y
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        """
        Copies the parameters from source network to target network
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def process_obs(self, obs):
        obs = np.array(obs, dtype='float32')
        obs = np.expand_dims(obs, axis=0)
        obs = torch.from_numpy(obs)
        return obs

    def process_action(self, actions):
        actions = torch.from_numpy(actions)
        return actions

    def process_reward(self, rewards):
        rewards = np.array(rewards, dtype='float32')
        rewards = torch.from_numpy(rewards)
        return rewards

    def process_done(self, done):
        done = np.array(done, dtype='float32')
        done = torch.from_numpy(done)
        return done

    def to_onehot(self, actions):
        # actions = np.argmax(actions, axis=-1)
        actions = np.array([np.random.choice(2, replace=False, p=x) for x in actions[0]])
        actions = np.expand_dims(actions, axis=0)
        actions = to_categorical(actions, num_classes=2)
        actions = actions.astype('float32')
        return actions

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        # state = np.expand_dims(state, axis=0)
        state = state.to(self.device)
        actions, _ = self.actor.forward(state, noise_inject=arglist.is_training)
        actions = actions.detach()
        actions = actions.data.cpu().numpy()
        # OU process: (-1, 1) scale
        # actions[:, :, 0:2] = actions[:, :, 0:2] + self.eps * self.noise.noise()
        actions[:, :, 0:2] = np.clip(actions[:, :, 0:2], a_min=-1, a_max=1)

        actions[:, :, 2:] = self.to_onehot(actions[:, :, 2:])
        # masking dead units
        actions = self._maskingActions(state, actions)
        # self.eps *= 0.99999

        return actions

    def _maskingActions(self, state, actions):
        mask = state[:, :, 1] == 0
        if type(actions).__module__ == 'torch':
            actions[mask] = 0
        elif type(actions).__module__ == 'numpy':
            mask = mask.data.cpu().numpy()
            mask = mask == 1
            actions[mask] = 0
        else:
            print('Not supported type!')
        return actions

    def _maskingPredState(self, state0, state1):
        mask = state0[:, :, 1] == 0
        state1[mask] = 0
        return state1


    def process_batch(self, experiences):
        s0 = torch.cat([e.state0[0] for e in experiences], dim=0)
        a0 = torch.cat([e.action for e in experiences], dim=0)
        r = torch.stack([e.reward for e in experiences], dim=0)
        s1 = torch.cat([e.state1[0] for e in experiences], dim=0)
        d = torch.stack([e.terminal1 for e in experiences], dim=0)

        return s0, a0, r, s1, d

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        experiences = self.memory.sample(arglist.batch_size)
        s0, a0, r, s1, d = self.process_batch(experiences)

        s0 = s0.to(self.device)
        a0 = a0.to(self.device)
        r = r.to(self.device)
        s1 = s1.to(self.device)
        d = d.to(self.device)

        self.actor.train()
        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a1, _ = self.target_actor.forward(s1)
        a1 = a1.detach()
        # masking dead units
        a1 = self._maskingActions(s1, a1)

        # target critic (1)
        q_next1, _ = self.target_critic1.forward(s1, a1)
        q_next1 = q_next1.detach()
        q_next1 = torch.squeeze(q_next1)
        # target critic (2)
        q_next2, _ = self.target_critic2.forward(s1, a1)
        q_next2 = q_next2.detach()
        q_next2 = torch.squeeze(q_next2)

        # min(Q1, Q2)
        q_next = torch.min(q_next1, q_next2)

        # Loss: TD error
        # y_exp = r + gamma*Q'( s1, pi'(s1))
        y_expected = r + GAMMA * q_next * (1. - d)
        # y_pred = Q( s0, a0)
        y_predicted, pred_r = self.critic.forward(s0, a0)
        y_predicted = torch.squeeze(y_predicted)
        pred_r = torch.squeeze(pred_r)

        # Sum. Loss
        critic_TDLoss = torch.nn.SmoothL1Loss()(y_predicted, y_expected)
        critic_ModelLoss = torch.nn.L1Loss()(pred_r, r)
        loss_critic = critic_TDLoss
        loss_critic += critic_ModelLoss * 0.1

        # Update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a0, pred_s1 = self.actor.forward(s0)
        # masking dead units
        pred_a0 = self._maskingActions(s0, pred_a0)

        # Loss: entropy for exploration
        entropy = torch.sum(pred_a0[:, :, 2:] * torch.log(pred_a0[:, :, 2:]), dim=-1)
        entropy[torch.isnan(entropy)] = 0
        entropy = entropy.mean()

        # Loss: regularization
        l2_reg = torch.cuda.FloatTensor(1)
        for W in self.actor.parameters():
            l2_reg = l2_reg + W.norm(2)

        # Loss: max. Q
        Q, _ = self.critic.forward(s0, pred_a0)
        actor_maxQ = -1 * Q.mean()

        # Loss: env loss
        # discard dead units
        s1 = self._maskingPredState(state0=s0, state1=s1)
        pred_s1 = self._maskingPredState(state0=s0, state1=pred_s1)
        actor_ModelLoss = torch.nn.L1Loss()(pred_s1, s1)

        # Sum. Loss
        loss_actor = actor_maxQ
        loss_actor += entropy * 0.05  # <replace Gaussian noise>
        loss_actor += torch.squeeze(l2_reg) * 0.001
        loss_actor += actor_ModelLoss * 0.1

        # Update actor
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
        self.actor_optimizer.step()

        # Update target env
        self.soft_update(self.target_actor, self.actor, arglist.tau)

        if self.n_updates % 2 == 0:
            self.soft_update(self.target_critic2, self.critic, arglist.tau)
        else:
            self.soft_update(self.target_critic1, self.critic, arglist.tau)
        self.n_updates += 1

        return loss_actor, loss_critic, critic_TDLoss, critic_ModelLoss, actor_maxQ, actor_ModelLoss

    def save_models(self):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), 'actor.pt')
        torch.save(self.target_critic1.state_dict(), 'critic1.pt')
        torch.save(self.target_critic2.state_dict(), 'critic2.pt')
        print('Models saved successfully')

    def load_models(self):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load('actor.pt'))
        self.hard_update(self.target_actor, self.actor)

        self.critic.load_state_dict(torch.load('critic1.pt'))
        self.hard_update(self.target_critic1, self.critic)

        self.critic.load_state_dict(torch.load('critic2.pt'))
        self.hard_update(self.target_critic2, self.critic)

        print('Models loaded succesfully')

    def save_training_checkpoint(self, state, is_best, episode_count):
        """
        Saves the models, with all training parameters intact
        :param state:
        :param is_best:
        :param filename:
        :return:
        """
        filename = str(episode_count) + 'checkpoint.path.rar'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')