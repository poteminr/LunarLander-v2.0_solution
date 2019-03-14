import copy
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

class DQN_agent(object):
    """
    Класс реализующий DQN агента
    """
    def __init__(self,
                 MODEL,
                 MemClass,
                 POLICY,
                 gamma=0.99,
                 alpha=1e-3,
                 maxMemorySize=1500,
                 tau=1e-3,
                 action_space=4,
                 observation_space=8,
                 seed=1
                 ):
        """
        MODEL - класс модели
        MemClass - класс памяти
        POLICY - класс для выбора действий
        action_space - количество доступных действий в среде
        observation_space - размер observation
        tau - коеф обновления весов закрепленной нейронки
        alpha - learning-rate
        ----------------------------------
        Q_nn_base - закрепленная нейронка
        Q_nn_copy - основная
        """
        self.POLICY = POLICY
        self.GAMMA = gamma
        self.ALPHA = alpha
        self.action_space = action_space
        self.observation_space = observation_space
        self.memSize = maxMemorySize
        self.TAU = tau
        self.learn_step_counter = 0
        self.memory = MemClass(self.memSize, observation_space, 1, 1, 228)
        torch.manual_seed(seed)

        self.Q_nn_copy = MODEL(alpha, observation_space, action_space)
        self.Q_nn_base = MODEL(alpha, observation_space, action_space)
        self.Q_nn_base.load_state_dict( self.Q_nn_copy.state_dict() ) # Одинаковые веса в начале
        self.Q_nn_base.eval()

    def chooseAction(self, observation):

        action = self.POLICY.action()

        if action is None:
            # make prediction

            self.Q_nn_copy.eval()

            with torch.no_grad():
                actions = self.Q_nn_copy(observation)

            return torch.argmax(actions).item()
        else:
            return action

    def learn(self, batch_size):

        observation, actions, rewards, next_observation, dones = self.memory.sample(batch_size)

        Q_base_predict = self.Q_nn_base(
            next_observation
            ).detach().max(1)[0].unsqueeze(1) # Максималная награда за действие

        y = rewards + ( self.GAMMA * Q_base_predict * (1-dones) )

        Q_copy_predict = self.Q_nn_copy(
            observation
            ).gather(1, actions) #Выбирает ревард того действия, которое было сделано в истории

        self.Q_nn_copy.optimizer.zero_grad()
        loss = self.Q_nn_copy.loss(
            Q_copy_predict,
            y
            )

        loss.backward()
        self.Q_nn_copy.optimizer.step()

        self.update_weight(self.Q_nn_copy, self.Q_nn_base, self.TAU)

    def update_weight(self, model_from, model_to, tau):
        for from_p, to_p in zip(model_from.parameters(), model_to.parameters()):
            to_p.data.copy_(tau*from_p.data + (1.0-tau)*to_p.data)
