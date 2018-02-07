import random
from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np

from cars.utils import Action
from learning_algorithms.network import Network
import learning_algorithms.network
from cars.Learner import Learner

class Agent(metaclass=ABCMeta):
    @property
    @abstractmethod
    def rays(self):
        pass

    @abstractmethod
    def choose_action(self, sensor_info):
        pass

class SimpleCarAgent(Agent):
    def __init__(self, n_rays, learner):
        """
        Создаёт машинку
        """
        self.evaluate_mode = False  # этот агент учится или экзаменутеся? если учится, то False
        self._rays = n_rays
        self.learner = learner

    @classmethod
    def from_file(cls, fname):
        nn = learning_algorithms.network.from_file(fname)
        n_rays = nn.weights[0].shape[1] - 4
        learner = Learner(n_rays)
        learner.neural_net = nn
        return SimpleCarAgent(n_rays, learner)

    @property
    def rays(self):
        return self._rays

    def choose_action(self, sensor_info):
        choose_random = (not self.evaluate_mode) and (random.random() < 0.05)
        (action, reward) = self.best_action_and_reward(sensor_info, choose_random=choose_random)
        self.learner.remember_history(sensor_info, action)
        return action

    # what after gamma:
    # max{a} Q(s_{t+1}, a)
    def estimate_of_optimal(self, sensor_info):
        (action, reward) = self.best_action_and_reward(sensor_info, choose_random=False)
        return reward

    def best_action_and_reward(self, sensor_info, choose_random):
        # хотим предсказать награду за все действия, доступные из текущего состояния
        rewards_to_controls_map = {}
        # дискретизируем множество значений, так как все возможные мы точно предсказать не сможем
        for steering in np.linspace(-1, 1, 3):  # выбирать можно и другую частоту дискретизации, но
            for acceleration in np.linspace(-0.75, 0.75, 3):  # в наших тестах будет именно такая
                action = Action(steering, acceleration)
                predicted_reward = self.learner.predict_reward(sensor_info, action)
                rewards_to_controls_map[predicted_reward] = action

        # ищем действие, которое обещает максимальную награду
        rewards = list(rewards_to_controls_map.keys())
        highest_reward = max(rewards)
        best_action = rewards_to_controls_map[highest_reward]

        # Добавим случайности, дух авантюризма. Иногда выбираем совершенно
        # рандомное действие
        if choose_random:
            highest_reward = rewards[np.random.choice(len(rewards))]
            best_action = rewards_to_controls_map[highest_reward]
        # следующие строки помогут вам понять, что предсказывает наша сеть
        #     print("Chosen random action w/reward: {}".format(highest_reward))
        # else:
        #     print("Chosen action w/reward: {}".format(highest_reward))

        return (best_action, highest_reward)
