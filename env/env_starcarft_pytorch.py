# -*- coding: utf-8 -*-
import numpy as np
import msgpack
from main import GlobalVariable as gvar
from util.CustomLog import cLogger
import time
from copy import deepcopy
import math

logger = cLogger.getLogger()

# env_details = {'ally': ['verture']*2,
#                'enemy': ['zealot']*3, 
#                'state_dim': (64, 64, 3 + 2)}


class StarCraftEnvironment(object):
    """Add class docstring."""
    reward_range = (-np.inf, np.inf)
    observation_space = None

    def __init__(self, agent_name, env_details):
        # intialize
        self.prev_health_ally = 0
        self.prev_health_enemy = 0

        self.agent_name = agent_name
        self.env_details = env_details
        self.num_ally = len(self.env_details['ally'])
        self.num_enemy = len(self.env_details['enemy'])
        self.state_dim = self.env_details['state_dim']
        self.nb_agents = self.num_ally

        # restart flag
        self.flag_restart = 0
        self.prev_flag_restart = False
        self.dummy_action = [[0, 0, 0]] * self.nb_agents

        # defalut health
        self.default_health_ally = 80 * len(self.env_details['ally'])
        self.default_health_enemy = 160 * len(self.env_details['enemy'])
        self.nb_step = 0
        self.R = 0
        self.nb_episode = 0

        # scenario information
        self.attack_range_of_ally = 5

        # log = open('results/train_step_log.txt', 'w')
        # log.write('train start... \n')
        # log.close()
        #
        # log = open('results/train_episode_log.txt', 'w')
        # log.write('train start... \n')
        # log.close()

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        [get_next_state in StarCraftEnvironment]
        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        # add step
        self.nb_step += 1

        # get restart flag
        self.flag_restart = int(gvar.flag_restart)
        # set action
        action_bwapi = self._make_action_bwapi(action)
        action_token = self._make_action_token(action_bwapi)
        gvar.release_action = True
        gvar.action = action_token

        # a -> s, r, d
        next_state = self._process_token()
        reward = self._get_reward()
        done = self._get_done()
        info = dict()

        if done:
            self.nb_episode += 1
            # np.save('results/hist/' + str(self.nb_episode) + '_reward.npy', self.R)

        # save step log
        # hp_a, hp_e = self._get_Health(self.token_unit)
        # msg = "episode: {}, step: {}, reward: {}, done: {}, allyHP: {}, enemyHP: {}".format(
        #             self.nb_episode,
        #             self.nb_step,
        #             round(reward, 3),
        #             done,
        #             hp_a,
        #             hp_e
        #             )
        # log = open('results/train_step_log.txt', 'a')
        # log.write(msg + '\n')
        # log.close()
        #
        # # save episode log
        # if done:
        #     self.nb_episode += 1
        #     msg = "episode: {}, step: {}, reward: {}, done: {}".format(
        #                 self.nb_episode,
        #                 self.nb_step,
        #                 round(self.R, 3),
        #                 done)
        #     log = open('results/train_episode_log.txt', 'a')
        #     log.write(msg + '\n')
        #     log.close()

        return next_state, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        [get_initial_state in StarCraftEnvironment]
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        self.nb_step = 0
        initial_state = self._process_token()
        self.R = []

        self.prev_health_ally = 80 * len(self.env_details['ally'])
        self.prev_health_enemy = 160 * len(self.env_details['enemy'])

        self.prev_num_ally = len(self.env_details['ally'])
        self.prev_num_enemy = len(self.env_details['enemy'])

        # remember unit id
        self.unit_id = self.token_unit[:, [0, 7]]

        return initial_state

    def _make_action_bwapi(self, action):
        # split action
        action_cont = action[:, :, 0:2]
        action_desc = action[:, :, 2:]

        action_bwapi = []
        # for each agent
        # print('actions')
        # print(self.unit_id[self.unit_id[:, 0] == 0, 1])
        for a_xy, a_type in zip(action_cont.squeeze(), action_desc.squeeze()):
            a_x = int(a_xy[0] * 128)
            a_y = int(a_xy[1] * 128)
            # [x, y, nothing/attack/move]
            if a_type[0] == a_type[1]:
                a_type = int(2)
            else:
                a_type = int(np.argmax(a_type))

            if a_type != 2:
                a = [a_x, a_y, a_type]
                # print(a)
                action_bwapi.append(a)

        return action_bwapi

    def _make_action_token(self, action_bwapi):
        action_token = msgpack.packb(action_bwapi)
        return action_token

    def _get_token(self):
        while True:
            if len(gvar.token_deque) == 0:
                time.sleep(1e-2)
                continue

            token = gvar.token_deque.pop()
            token = msgpack.unpackb(token, raw=False)
            if len(token) == 1:
                # there are no units! Send dummy action
                action_token = self._make_action_token(self.dummy_action)
                gvar.release_action = True
                gvar.action = action_token
            else:
                #logger.info('state\n' + str(token))
                break
        return token

    def _parse_token(self, token):
        if len(token) == 1:
            token_unit = 0
            token_resource = token[-1]
        else:
            token_unit = token[:-1]
            token_resource = token[-1]
        return token_unit, token_resource

    def _str2num(self, token_unit, token_resource):
        # make numpy array
        token_unit = np.array([np.array(x, dtype='float32') for x in token_unit])
        token_resource = np.array(token_resource, dtype='float32')
        return token_unit, token_resource

    def _calibrate_token_unit(self, token_unit):
        # no units
        if len(token_unit.shape) == 1:
            token_unit = np.zeros(shape=(self.num_ally + self.num_enemy, 11))
            token_unit[:self.num_ally, 0] = 0  # assign ally
            token_unit[self.num_ally:, 0] = 1  # assign enemy
            return token_unit

        elif len(token_unit) == (self.num_ally + self.num_enemy):
            return token_unit

        else:
            # when units are dead
            # current units
            token_unit_empty = np.zeros(shape=(self.num_ally + self.num_enemy, 11))
            token_unit_empty[:, [0, 7]] = self.unit_id

            for token_unit_single in token_unit:
                idx = token_unit_single[7]
                token_unit_empty[token_unit_empty[:, 7] == idx] = token_unit_single

            token_unit = token_unit_empty
            return token_unit

    def _make_observation(self, token_unit):
        '''
        <input info>
        BWAPI: for a unit, vector = [isEnemy, HP, Sheild, Cooldown, X, Y, UnitType, ID, isAttacking, isMoving, isUnderAttack]
        token_unit = np.array([unit1, unit2, ..., unitN])
        unitN = np.array([isEnemy, HP, Sheild, Cooldown, X, Y, UnitType, ID, isAttacking, isMoving, isUnderAttack])

        <output info>
        out = np.array([unit1, unit2, ..., unitN])
        unitN = (2D np.arrray, 1D np.array)
        '''
        token_unit[:, 4:6] = token_unit[:, 4:6] / 128.  # ?? scale
        token_unit = np.delete(token_unit, [6, 7], axis=1)

        token_unit_ally = token_unit[token_unit[:, 0] == 0]
        token_unit_enemy = token_unit[token_unit[:, 0] == 1]

        token_unit_enemy = token_unit_enemy.flatten()
        observation = []
        for ally in token_unit_ally:
            observation.append(np.concatenate([ally, token_unit_enemy]))
        observation = np.array(observation)

        return observation

    def _process_token(self):
        token = self._get_token()
        token_unit, token_resources = self._parse_token(token)
        token_unit, token_resources = self._str2num(token_unit, token_resources)

        # unit state
        # check dead ally to calibrate state
        token_unit = self._calibrate_token_unit(token_unit)

        # to use for reward
        # resource: mineral, gas
        self.token_unit = deepcopy(token_unit)
        self.mineral = token_resources[0]  # win count
        self.gas = token_resources[1]  # game count

        # make observation
        obs = self._make_observation(token_unit)

        return obs

    def _get_Health(self, token_unit):
        # isEnemysf
        ally = token_unit[token_unit[:, 0] == 0]
        enemy = token_unit[token_unit[:, 0] == 1]

        Health_ally = sum(ally[:, 1]) + sum(ally[:, 2])
        Health_enemy = sum(enemy[:, 1]) + sum(enemy[:, 2])

        return Health_ally, Health_enemy

    def _get_reward(self):
        """
        token_unit: [isEnemy, HP, Sheild, Cooldown, X, Y, UnitType, ID, isAttacking, isMoving, isUnderAttack]
        attack range of vulture : 5
        vulture v1_x, v1_y, v2_x, v2_y
        zealot z1_x, z1_y, z2_x, z2_y
        Health(hp(1) + shield(2)) change ratio of vulture (delta_ally)
        Health(hp(1) + shield(2)) change ratio of zealot (delta_enemy)
        dead_count_of_vulture
        dead_count_of_zealot
        attacking status of vulture = attacking_1, attacking_2
        status of vulture under attach = under_attack_1, under_attack_2
        """
        token_unit_ally = self.token_unit[self.token_unit[:, 0] == 0, :]
        token_unit_enemy = self.token_unit[self.token_unit[:, 0] == 1, :]

        # get health
        currentHealth_ally, currentHealth_enemy = self._get_Health(self.token_unit)

        # reward by health change
        delta_ally = self.prev_health_ally - currentHealth_ally
        delta_enemy = self.prev_health_enemy - currentHealth_enemy

        # scaling delta
        delta_ally = delta_ally / self.default_health_ally
        delta_enemy = delta_enemy / self.default_health_enemy

        # units count alive
        hp_ally = token_unit_ally[:, 1] + token_unit_ally[:, 2]
        hp_enemy = token_unit_enemy[:, 1] + token_unit_enemy[:, 2]

        token_unit_ally = token_unit_ally[hp_ally > 0].reshape((-1, 11))
        token_unit_enemy = token_unit_enemy[hp_enemy > 0].reshape((-1, 11))

        # (x, y)
        pos_ally = token_unit_ally[:, 4:6] / 32
        pos_enemy = token_unit_enemy[:, 4:6] / 32

        # num units dead
        num_dead_ally = self.prev_num_ally - len(token_unit_ally)
        num_dead_enemy = self.prev_num_enemy - len(token_unit_enemy)
        '''
        # pairwise distance
        Ds = []
        for a in pos_ally:
            for b in pos_enemy:
                d = self._dist(a, b)  # tile position scale
                Ds.append(d)

        # calc. reward by min dis. (-5, 0)
        range_reward = 0
        if len(Ds) != 0:
            d = min(Ds)
            if d <= 5:
                range_reward += -abs(d - 5)
            elif d > 5:
                range_reward += -(5 / 128) * abs(d - 5)
        '''
        # attacking_status_of_vulture & status of vulture under attack
        is_attack = token_unit_ally[:, 8]
        is_underattack = token_unit_ally[:, 9]
        is_control = (is_attack + is_underattack) == 0
        reward_contol = sum(is_control)

        reward = 0
        # 1. reward by minimum distance
        # reward += range_reward

        # 2. isAttacking and underAttack handling (-4, 4)
        # reward += (is_attack - is_underattack) / 2.

        # 3. change ratio hp (-1, 1)
        reward += -delta_ally + delta_enemy

        # 4. dead unit handling (-4, 3)
        # reward += (-0.4 * num_dead_ally + 0.6 * num_dead_enemy)
        # reward += - num_dead_ally

        # 5. control penalty
        # reward += -reward_contol / 4

        # update hp previous
        self.prev_health_ally = currentHealth_ally
        self.prev_health_enemy = currentHealth_enemy

        self.prev_num_ally = len(token_unit_ally)
        self.prev_num_enemy = len(token_unit_enemy)

        # ## for debug
        self.R.append(reward)
        #
        # self.hist.append(np.array([self.nb_step, self.prev_health_ally, currentHealth_ally,
        #                            self.prev_health_enemy, currentHealth_enemy, reward, self.R]))
        # reward = 1
        return reward

    def _dist(self, a, b):
        '''
        a = np.array([0, 0])
        b = np.array([1, 1])
        '''
        return sum((a - b) ** 2) ** 0.5

    def _get_done(self):
        done = False

        ally = self.token_unit[self.token_unit[:, 0] == 0]
        enemy = self.token_unit[self.token_unit[:, 0] == 1]

        # self.currentHealth_ally = sum(ally[:, 1])
        # self.currentHealth_enemy = sum(enemy[:, 1]) + sum(enemy[:, 2])

        # test1 = (self.currentHealth_ally - self.prev_health_ally) >= (self.default_health_ally - 8)
        num_ally = sum(ally[:, 1] > 0)
        num_enemy = sum(enemy[:, 1] > 0)

        if (num_ally == 0 or num_enemy == 0) and self.nb_step > 10:
            done = True

        if self.flag_restart == 1:
            done = True

        elif self.flag_restart == 2:
            done = True

        # update prev_health_ally
        # self.prev_health_ally = self.currentHealth_ally
        # self.prev_health_enemy = self.currentHealth_enemy

        return done
