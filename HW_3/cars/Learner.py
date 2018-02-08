from collections import deque, namedtuple
from learning_algorithms.network import Network
import numpy as np

HistoryItem = namedtuple('HistoryItem', ['state', 'action', 'qvalue', 'reward'])

class Learner:

    def __init__(self, n_rays):
        self.evaluate_mode = False
        # here +2 is for 2 inputs from elements of Action that we are trying to predict
        self.neural_net = Network([n_rays + 4,
                                   # внутренние слои сети: выберите, сколько и в каком соотношении вам нужно
                                   # например, (self.rays + 4) * 2 или просто число
                                   n_rays + 4,
                                   (n_rays + 4) // 2,
                                   1],
                                  output_function=lambda x: x, output_derivative=lambda x: 1)
        self.ALPHA = 0.1
        self.GAMMA = 0.8

        self.reset_history()

    def reset_history(self):
        self.history = deque([], maxlen=50000)

        self.step = 0

        self.last_reward = 0
        self.last_qvalue = 0

    def start_episode(self):
        self.reset_history()

    def update_qvalue(self, state, action, reward, agent, next_agent_state):
        old_qvalue = self.predict_reward(state, action)
        item = HistoryItem(state, action, old_qvalue, reward)
        self.history.append(item)
        self.last_reward = reward

    def update_final_qvalue(self, state, action, reward):
        item = HistoryItem(state, action, reward, reward)
        self.history.append(item)
        self.last_reward = reward

    def backtrack_qvalues(self, history, callback):
        qvalue = None
        for item in (list(history))[::-1]:
            if qvalue is None:
                qvalue = item.reward
                print("qvalue in final state:", qvalue)
            else:
                qvalue = self.calculate_new_qvalue(item.qvalue, item.reward, qvalue)
            callback(item, qvalue)

    def calculate_new_qvalue(self, old_qvalue, reward, estimate_of_optimal):
        new_qvalue = (1 - self.ALPHA) * old_qvalue + self.ALPHA * (reward + self.GAMMA * estimate_of_optimal)
        print("Q value update. New: %.4f, old: %.4f, reward: %.4f, estimate: %.4f" % (new_qvalue, old_qvalue, reward, estimate_of_optimal))
        return new_qvalue

    def learn(self):
        def tuple_to_ndvector(x):
            v = np.asarray(x)
            v = v.flatten()[:, np.newaxis]
            return v
        training_data = []
        def on_training_item(item, qvalue):
            x = tuple_to_ndvector(self.state_and_action_to_neunet_vector(item.state, item.action))
            training_data.append((x, qvalue))
        self.backtrack_qvalues(self.history, on_training_item)
        self.neural_net.SGD(training_data=training_data, epochs=15, mini_batch_size=50, eta=0.05)

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
        agent_vector_representation = self.state_and_action_to_neunet_vector(state, action)
        return float(self.neural_net.feedforward(agent_vector_representation))

    # Vector is horizontal
    def state_and_action_to_vector(self, state, action):
        agent_vector_representation = np.append(state, action)
        # It would make the vector vertical
        #agent_vector_representation = agent_vector_representation.flatten()[:, np.newaxis]
        return tuple(agent_vector_representation)

    def state_and_action_to_neunet_vector(self, state, action):
        agent_vector_representation = np.append(state, action)
        agent_vector_representation = agent_vector_representation.flatten()[:, np.newaxis]
        return agent_vector_representation

