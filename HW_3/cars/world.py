import itertools
import random
from abc import ABCMeta, abstractmethod
from cmath import rect, pi, phase
from time import sleep

import numpy as np
import pygame

import gym

if __name__ == "__main__":
    import sys
    sys.path.append('..')
from cars.agent import SimpleCarAgent
from cars.track import plot_map
from cars.physics import SimplePhysics
from cars.utils import CarState, to_px, rotate, intersect_ray_with_segment, draw_text, angle, Action
from cars.Learner import Learner

import learning_algorithms.network

black = (0, 0, 0)
white = (255, 255, 255)

class ActionSpace(gym.Space):
    def __init__(self):
        self.actions = [Action(steering, acceleration)
                for steering in np.linspace(-1, 1, 3)
                for acceleration in np.linspace(0.75, 0.75, 3)]

    def sample(self):
        i = random.randint(0, len(self.actions)-1)
        return self.actions[i]

class SimpleCarWorld(gym.Env):
    COLLISION_PENALTY =  3 # выберите сами
    HEADING_REWARD =  0 # выберите сами
    WRONG_HEADING_PENALTY = 1  # выберите сами
    IDLENESS_PENALTY = 1 # выберите сами
    SPEEDING_PENALTY = 1  # выберите сами
    MIN_SPEED = 0.1 # выберите сами
    MAX_SPEED = 5 # выберите сами
    N_RAYS = 7 # Number of rays

    size = (800, 600)

    def __init__(self, car_map):
        """
        Инициализирует мир
        :param car_map: карта, на которой всё происходит (см. track.py0
        :param physics_pars: дополнительные параметры, передаваемые в конструктор класса физики
        (кроме car_map, являющейся обязательным параметром конструктора)
        """
        self.physics = SimplePhysics(car_map, timedelta=0.2)
        self.map = car_map
        self.action_space = ActionSpace()
        self.done = False

        # создаём агентов

        self._info_surface = pygame.Surface(self.size)

        self._car_surfaces = []
        self._car_images = []

        self.reset_cars()

    def reset_cars(self):
        pos = (self.map[0][0] + self.map[0][1]) / 2
        vel = 0
        heading = rect(-0.3, 1)

        self.car_state = CarState(pos, vel, heading)
        self.circles = 0

    def step(self, action):
        """
        Логика основного цикла:
         подсчёт для каждого агента видения агентом мира,
         выбор действия агентом,
         смена состояния
         и обработка реакции мира на выбранное действие
        """
        """
        for a in self.cars:
            action = a.choose_action(vision) # a_t
            """
        next_car_state, collision = self.physics.move(
                self.car_state, action)
        angle_inc = angle(self.car_state.position,
                next_car_state.position) / (2*pi)
        self.circles += angle_inc
        vision = self.vision_for(self.car_state)
        return (vision, angle_inc, self.done, None)

    def reward(self, state, collision):
        """
        Вычисление награды агента, находящегося в состоянии state.
        Эту функцию можно (и иногда нужно!) менять, чтобы обучить вашу сеть именно тем вещам, которые вы от неё хотите
        :param state: текущее состояние агента
        :param collision: произошло ли столкновение со стеной на прошлом шаге
        :return reward: награду агента (возможно, отрицательную)
        """
        a = np.sin(angle(-state.position, state.heading))
        heading_reward = 1 if a > 0.1 else a if a > 0 else 0
        heading_penalty = -1 if a <= 0 else 0
        idle_penalty = 0 if abs(state.velocity) > self.MIN_SPEED else -self.IDLENESS_PENALTY
        speeding_penalty = 0 if abs(state.velocity) < self.MAX_SPEED else -self.SPEEDING_PENALTY
        collision_penalty = - int(collision) * self.COLLISION_PENALTY

        return heading_reward * self.HEADING_REWARD + heading_penalty * self.WRONG_HEADING_PENALTY + collision_penalty \
               + idle_penalty + speeding_penalty

    def eval_reward(self, state, collision):
        """
        Награда "по умолчанию", используется в режиме evaluate
        Удобно, чтобы не приходилось отменять свои изменения в функции reward для оценки результата
        """
        a = -np.sin(angle(-state.position, state.heading))
        heading_reward = 1 if a > 0.1 else a if a > 0 else 0
        heading_penalty = a if a <= 0 else 0
        idle_penalty = 0 if abs(state.velocity) > self.MIN_SPEED else -self.IDLENESS_PENALTY
        speeding_penalty = 0 if abs(state.velocity) < self.MAX_SPEED else -self.SPEEDING_PENALTY * abs(state.velocity)
        collision_penalty = - max(abs(state.velocity), 0.1) * int(collision) * self.COLLISION_PENALTY

        return heading_reward * self.HEADING_REWARD + heading_penalty * self.WRONG_HEADING_PENALTY + collision_penalty \
            + idle_penalty + speeding_penalty

    def run(self, n_episodes, n_steps=None):
        """
        Основной цикл мира; по завершении сохраняет текущие веса агента в файл network_config_car_n_layers_....txt
        :param n_steps: количество шагов цикла; до внешней остановки, если None
        """
        scale = self._prepare_visualization()
        done = False
        for i_episode in range(n_episodes) if n_episodes is not None else itertools.count():
            print("Episode", i_episode)
            self.reset_cars()
            self.learner.start_episode()
            for _ in range(n_steps) if n_steps is not None else itertools.count():
                self.transition()
                self.visualize(scale)
                if self._update_display() == pygame.QUIT:
                    done = True
                    break
                sleep(0.1)
            if not done:
                self.learner.learn()
            if done:
                break

        filename = "network_config_learner_layers_%s.txt" % ("_".join(map(str, self.learner.neural_net.sizes)))
        learning_algorithms.network.to_file(self.learner.neural_net, filename)
        print("Saved neural network parameters to '%s'" % filename)

    def evaluate_car(self, car, steps=1000, visual=True):
        """
        Прогонка цикла мира для конкретного агента (см. пример использования в комментариях после if _name__ == "__main__")
        :param car: SimpleCarAgent
        :param steps: количество итераций цикла
        :param visual: рисовать картинку или нет
        :return: среднее значение награды агента за шаг
        """
        car.evaluate_mode = True
        self.set_cars([car])
        rewards = []
        if visual:
            scale = self._prepare_visualization()
        for _ in range(steps):
            vision = self.vision_for(car)
            action = car.choose_action(vision)
            next_car_state, collision = self.physics.move(
                self.car_states[car], action
            )
            self.circles[car] += angle(self.car_states[car].position, next_car_state.position) / (2*pi)
            self.car_states[car] = next_car_state
            rewards.append(self.eval_reward(next_car_state, collision))
            car.receive_feedback(rewards[-1])
            if visual:
                self.visualize(scale)
                if self._update_display() == pygame.QUIT:
                    break
                sleep(0.05)

        return np.mean(rewards)

    def vision_for(self, car_state):
        """
        Строит видение мира для каждого агента
        :param car: машинка, из которой мы смотрим
        :return: список из модуля скорости машинки, направленного угла между направлением машинки
        и направлением на центр и `car.rays` до ближайших стен трека (запустите картинку, и станет совсем понятно)
        :state: Force the position of the car, otherwise the current position is used
        """
        state = car_state
        vision = [abs(state.velocity), np.sin(angle(-state.position, state.heading))]
        extras = len(vision)

        delta = pi / (self.N_RAYS - 1)
        start = rotate(state.heading, - pi / 2)

        sectors = len(self.map)
        for i in range(self.N_RAYS):
            # define ray direction
            ray = rotate(start, i * delta)

            # define ray's intersections with walls
            vision.append(np.infty)
            for j in range(sectors):
                inner_wall = self.map[j - 1][0], self.map[j][0]
                outer_wall = self.map[j - 1][1], self.map[j][1]

                intersect = intersect_ray_with_segment((state.position, ray), inner_wall)
                intersect = abs(intersect - state.position) if intersect is not None else np.infty
                if intersect < vision[-1]:
                    vision[-1] = intersect

                intersect = intersect_ray_with_segment((state.position, ray), outer_wall)
                intersect = abs(intersect - state.position) if intersect is not None else np.infty
                if intersect < vision[-1]:
                    vision[-1] = intersect

            assert vision[-1] < np.infty, \
                "Something went wrong: {}, {}".format(str(state), str(car.chosen_actions_history[-1]))
        assert len(vision) == self.N_RAYS + extras, \
            "Something went wrong: {}, {}".format(str(state), str(car.chosen_actions_history[-1]))
        return vision

    def visualize(self, scale):
        """
        Рисует картинку. Этот и все "приватные" (начинающиеся с _) методы необязательны для разбора.
        """
        for i, car in enumerate(self.cars):
            state = self.car_states[car]
            surface = self._car_surfaces[i]
            rays_lengths = self.vision_for(car)[-self.N_RAYS:]
            self._car_images[i] = [self._draw_ladar(rays_lengths, state, scale),
                                     self._get_car_image(surface, state, scale)]

        if len(self.cars) == 1:
            a = self.cars[0]
            draw_text("Reward: %.3f, Q Value: %.3f" % (self.learner.last_reward, self.learner.last_qvalue), self._info_surface, scale, self.size,
                      text_color=white, bg_color=black)
            try:
                steer, acc = self.learner.history[-1].action
            except IndexError:
                steer = 0
                acc = 0
            state = self.car_states[a]
            draw_text("Action: steer.: %.2f, accel: %.2f" % (steer, acc), self._info_surface, scale,
                      self.size, text_color=white, bg_color=black, tlpoint=(self._info_surface.get_width() - 500, 10))
            draw_text("Inputs: |v|=%.2f, sin(angle): %.2f, circle: %.2f" % (
                abs(state.velocity), np.sin(angle(-state.position, state.heading)), self.circles[a]),
                      self._info_surface, scale,
                      self.size, text_color=white, bg_color=black, tlpoint=(self._info_surface.get_width() - 500, 50))

    def _get_car_image(self, original, state, scale):
        angle = phase(state.heading) * 180 / pi
        rotated = pygame.transform.rotate(original, angle)
        rectangle = rotated.get_rect()
        rectangle.center = to_px(state.position, scale, self.size)
        return rotated, rectangle

    def _draw_ladar(self, sensors, state, scale):
        surface = pygame.display.get_surface().copy()
        surface.fill(white)
        surface.set_colorkey(white)
        start_pos = to_px(state.position, scale, surface.get_size())
        delta = pi / (len(sensors) - 1)
        ray = phase(state.heading) - pi / 2
        for s in sensors:
            end_pos = to_px(rect(s, ray) + state.position, scale, surface.get_size())
            pygame.draw.line(surface, (0, 255, 0), start_pos, end_pos, 2)
            ray += delta

        rectangle = surface.get_rect()
        rectangle.topleft = (0, 0)
        return surface, rectangle

    def _prepare_visualization(self):
        red = (254, 0, 0)
        pygame.init()
        screen = pygame.display.set_mode(self.size)
        screen.fill(white)
        scale = plot_map(self.map, screen)
        for state in self.car_states.values():
            s = pygame.Surface((25, 15))
            s.set_colorkey(white)
            s.fill(white)
            pygame.draw.rect(s, red, pygame.Rect(0, 0, 15, 15))
            pygame.draw.polygon(s, red, [(15, 0), (25, 8), (15, 15)], 0)
            self._car_surfaces.append(s)
            self._car_images.append([self._get_car_image(s, state, scale)])

        self._map_surface = screen
        return scale

    def _update_display(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return pygame.QUIT
        display = pygame.display.get_surface()
        display.fill(white)

        plot_map(self.map, display)
        for images in self._car_images:
            for surf, rectangle in images:
                display.blit(surf, rectangle)
        display.blit(self._info_surface, (0, 0), None, pygame.BLEND_RGB_SUB)
        self._info_surface.fill(black)  # clear notifications from previous round
        pygame.display.update()


if __name__ == "__main__":
    from cars.physics import SimplePhysics
    from cars.track import generate_map

    np.random.seed(3)
    random.seed(3)
    m = generate_map(8, 5, 3, 3)
    env = SimpleCarWorld(m)
    print("Step:", env.step(env.action_space.sample()))
    print("Step:", env.step(env.action_space.sample()))
    print("Step:", env.step(env.action_space.sample()))
    print("Step:", env.step(env.action_space.sample()))
    print("Step:", env.step(env.action_space.sample()))

    # если вы хотите продолжить обучение уже существующей модели, вместо того,
    # чтобы создавать новый мир с новыми агентами, используйте код ниже:
    # # он загружает агента из файла
    # car = SimpleCarAgent.from_file('filename.txt')
    # # создаёт мир
    # w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2)
    # # подключает к нему агента
    # w.set_cars([car])
    # # и запускается
    # w.run()
    # # или оценивает агента в этом мире
    # print(w.evaluate_car(car, 500))
