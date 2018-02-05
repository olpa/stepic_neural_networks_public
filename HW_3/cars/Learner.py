from collections import deque
from learning_algorithms.network import Network
import numpy as np

class Learner:

    def __init__(self, n_rays, history_data=int(50000)):
        self.evaluate_mode = False
        self.sensor_data_history = deque([], maxlen=history_data)
        self.chosen_actions_history = deque([], maxlen=history_data)
        self.reward_history = deque([], maxlen=history_data)
        # here +2 is for 2 inputs from elements of Action that we are trying to predict
        self.neural_net = Network([n_rays + 4,
                                   # внутренние слои сети: выберите, сколько и в каком соотношении вам нужно
                                   # например, (self.rays + 4) * 2 или просто число
                                   1],
                                  output_function=lambda x: x, output_derivative=lambda x: 1)
        self.step = 0
        self.q_table = {}

    def remember_history(self, sensor_info, best_action):
        # запомним всё, что только можно: мы хотим учиться на своих ошибках
        self.sensor_data_history.append(sensor_info)
        self.chosen_actions_history.append(best_action)
        self.reward_history.append(0.0)  # мы пока не знаем, какая будет награда, это
        # откроется при вызове метода receive_feedback внешним миром

    def update_qvalue(self, state, action, reward, agent, next_agent_state):
        stac = self.state_and_action_to_vector(state, action)
        self.q_table[stac] = reward

    def update_final_qvalue(self, state, action, reward):
        stac = self.state_and_action_to_vector(state, action)
        self.q_table[stac] = reward

    def receive_feedback(self, reward, train_every=50, reward_depth=7):
        """
        Получить реакцию на последнее решение, принятое сетью, и проанализировать его
        :param reward: оценка внешним миром наших действий
        :param train_every: сколько нужно собрать наблюдений, прежде чем запустить обучение на несколько эпох
        :param reward_depth: на какую глубину по времени распространяется полученная награда
        """
        # считаем время жизни сети; помогает отмерять интервалы обучения
        self.step += 1

        # начиная с полной полученной истинной награды,
        # размажем её по предыдущим наблюдениям
        # чем дальше каждый раз домножая её на 1/2
        # (если мы врезались в стену - разумно наказывать не только последнее
        # действие, но и предшествующие)
        i = -1
        while len(self.reward_history) > abs(i) and abs(i) < reward_depth:
            self.reward_history[i] += reward
            reward *= 0.5
            i -= 1

        # Если у нас накопилось хоть чуть-чуть данных, давайте потренируем нейросеть
        # прежде чем собирать новые данные
        # (проверьте, что вы в принципе храните достаточно данных (параметр `history_data` в `__init__`),
        # чтобы условие len(self.reward_history) >= train_every выполнялось
        if not self.evaluate_mode and (len(self.reward_history) >= train_every) and not (self.step % train_every):
            X_train = np.concatenate([self.sensor_data_history, self.chosen_actions_history], axis=1)
            y_train = self.reward_history
            train_data = [(x[:, np.newaxis], y) for x, y in zip(X_train, y_train)]
            self.neural_net.SGD(training_data=train_data, epochs=15, mini_batch_size=train_every, eta=0.05)

    def predict_reward(self, state, action):
        agent_vector_representation = self.state_and_action_to_vector(state, action)
        return float(self.neural_net.feedforward(agent_vector_representation))

    # Vector is horizontal
    def state_and_action_to_vector(self, state, action):
        agent_vector_representation = np.append(state, action)
        # It would make the vector vertical
        #agent_vector_representation = agent_vector_representation.flatten()[:, np.newaxis]
        return tuple(agent_vector_representation)
