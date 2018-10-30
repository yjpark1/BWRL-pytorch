# -*- coding: utf-8 -*-
import numpy as np
import msgpack
from main import GlobalVariable as gvar
from util.CustomLog import cLogger

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
        gvar.release_action = False
        gvar.action = action_token

        # a -> s, r, d
        next_state = self._process_token()
        reward = self._get_reward()
        done = self._get_done()
        info = dict()

        # stop rl agent
        gvar.service_flag = 0

        return next_state, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        [get_initial_state in StarCraftEnvironment]
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        gvar.release_action = True # ????
        initial_state = self._process_token()

        return initial_state

    def _make_action_bwapi(self, action):
        action_bwapi = []
        # for each agent
        for a_xy, a_type in zip(action[0], action[1]):
            a_xy = np.where(np.arange(64 * 64).reshape(64, 64) == a_xy)

            # use reference coordinate
            a_xy = self.reference_coor[a_xy[0], a_xy[1], :]
            a_xy = np.squeeze(a_xy)

            a_x = int(a_xy[0] * 4)
            a_y = int(a_xy[1] * 4)
            # [x, y, nothing/attack/move]
            a_type = int(a_type)
            a = [a_x, a_y, a_type]
            action_bwapi.append(a)
        return action_bwapi

    def _make_action_token(self, action_bwapi):
        action_token = msgpack.packb(action_bwapi)
        return action_token

    def _get_token(self):
        while True:
            if len(gvar.token_deque) == 0:
                continue

            gvar.release_action = True
            token = gvar.token_deque.pop()
            token = msgpack.unpackb(token, raw=False)
            if len(token) == 1:
                # there are no units! Send dummy action
                action_token = self._make_action_token(self.dummy_action)
                gvar.action = action_token
            else:
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
        if token_unit == 0:
            token_unit = np.zeros(shape=(self.num_ally + self.num_enemy, 11))
            token_unit[:self.num_ally, 0] = 0  # assign ally
            token_unit[self.num_ally:, 0] = 1  # assign enemy

        # when units are dead
        # current units
        numAlly = sum(self.token_unit[:, 0] == 0)
        numEnemy = sum(self.token_unit[:, 0] == 1)

        # dead units
        numDeadAlly = self.num_ally - numAlly
        numDeadEnemy = self.num_enemy - numEnemy

        # calibration
        if numDeadAlly > 0:
            add = np.zeros((numDeadAlly, 11))
            token_unit = np.vstack([token_unit, add])

        if numDeadEnemy > 0:
            add = np.zeros((numDeadEnemy, 11))
            add[:, 0] = 1
            token_unit = np.vstack([token_unit, add])

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
        token_unit[:, 4:6] = token_unit[:, 4:6] / (64*8)  # ?? scale
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
        obs = self._make_observation(token_unit)

        # to use for reward
        # resource: mineral, gas
        self.token_unit = token_unit
        self.mineral = token_resources[0]  # win count
        self.gas = token_resources[1]  # game count

        return obs

    def _get_Health(self):
        # isEnemy
        ally = self.token_unit[self.token_unit[:, 0] == 0]
        enemy = self.token_unit[self.token_unit[:, 0] == 1]

        self.currentHealth_ally = sum(ally[:, 1])
        self.currentHealth_enemy = sum(enemy[:, 1]) + sum(enemy[:, 2])

    # _get_reward 함수와 _get_done 함수는 서로 얽혀 있음. 수정 시 주의요
    def _get_reward(self):
        '''
        health = hp(1) + sheild(2)
        '''
        # get health
        self._get_Health()
        currentHealth_ally = self.currentHealth_ally
        currentHealth_enemy = self.currentHealth_enemy

        '''
        # reward by health change
        delta_ally = self.prev_health_ally - currentHealth_ally
        delta_enemy = self.prev_health_enemy - currentHealth_enemy

        # scaling delta
        delta_ally = 10 * delta_ally / self.default_health_ally
        delta_enemy = 10 * delta_enemy / self.default_health_enemy

        reward = delta_enemy - delta_ally
        '''

        # reward by steps
        reward = 0.5 #  ( 0 + 1 ) / 2 = 0.5

        '''
        # reward by remain health
        delta_ally = self.default_health_ally - currentHealth_ally
        delta_enemy = self.default_health_enemy - currentHealth_enemy

        test_delta_ally = self.prev_health_ally != currentHealth_ally
        test_delta_enemy = self.prev_health_enemy != currentHealth_enemy

        reward = 0
        if test_delta_ally or test_delta_enemy:
            delta_ally = self.default_health_ally - currentHealth_ally
            delta_enemy = self.default_health_enemy - currentHealth_enemy

            # scaling delta
            delta_ally = 1 * delta_ally / self.default_health_ally
            delta_enemy = 1 * delta_enemy / self.default_health_enemy

            reward = delta_enemy - delta_ally
        '''
        '''
        if currentHealth_ally == 0 or currentHealth_enemy == 0:
            # 다 끝난 경우
            reward = 0

        elif (currentHealth_ally == self.default_health_ally and
              currentHealth_enemy == self.default_health_enemy):
            # 새로 시작한 경우
            reward = 0
        '''

        ## for debug
        self.R += reward

        '''
        self.hist.append(np.array([self.nb_step, self.prev_health_ally, currentHealth_ally,
                                   self.prev_health_enemy, currentHealth_enemy, reward, self.R]))

        if self.R < -100:
            hist = np.array(self.hist)
            print(hist)
            raise ArithmeticError
        '''
        ##
        # print(self.token_unit)
        # print('ally HP:', currentHealth_ally, ' enemy HP: ', currentHealth_enemy, ' reward: ', reward, ' cumulative reward: ', self.R)
        ##

        return reward

    def _get_done(self):
        done = False
        currentHealth_ally = self.currentHealth_ally
        currentHealth_enemy = self.currentHealth_enemy

        '''
        test1 = self.currentHealth_ally - self.prev_health_ally == self.default_health_ally
        test2 = self.currentHealth_enemy - self.prev_health_enemy == self.default_health_enemy
        '''
        test1 = (self.currentHealth_ally - self.prev_health_ally) >= (self.default_health_ally - 8)
        test2 = (self.currentHealth_enemy - self.prev_health_enemy) >= (self.default_health_enemy - 8)

        if (test1 or test2) and self.nb_step > 10:
            done = True

        '''
        print("self.flag_restart: ", self.flag_restart, " self.prev_flat_restart: ", self.prev_flag_restart)

        if (not self.prev_flag_restart) and (self.flag_restart == 1):
            done = True

        elif (test1 or test2) and self.nb_step > 10:
            done = True

        self.prev_flag_restart = self.flag_restart
        '''
        # update prev_health_ally
        self.prev_health_ally = currentHealth_ally
        self.prev_health_enemy = currentHealth_enemy

        return done

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        pass

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

