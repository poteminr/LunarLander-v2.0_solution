import numpy as np
import copy

class BasePolicy():
    def __init__(self,
                 eps=1.,
                 how="multiplic",
                 min_eps=0.05,
                 eps_delta=0.995,
                 action_space=4
                 ):
        """
        eps - коеф рандома
        min_eps - минимальный epsilon
        eps_delta - коеф уменьшения epsilonа
        how - каким образом именьшать
        action_space - количество доступных действий в среде
        """
        self.EPS = eps
        self.MIN_EPS = min_eps
        self.EPS_DELTA = eps_delta
        self.how = how
        self.action_space = action_space


    def _eps_minimize(self):

        if (self.how == "minus"):
            self.EPS = max(self.MIN_EPS, self.EPS-self.EPS_DELTA)
        elif (self.how == "multiplic"):
            self.EPS = max(self.MIN_EPS, self.EPS*self.EPS_DELTA)


    def action(self):
        """
        Возвращает None если действие выбирает сеть, иначе номер действия

        """
        rand = np.random.random()
        action = None

        if rand >= 1 - self.EPS:
            action = np.random.choice(range(self.action_space))

        self._eps_minimize()

        return action

class JumpPolicy():
    def __init__(self,
                 eps=1.,
                 min_eps=0.05,
                 eps_delta=0.995,
                 n_jumps=0,
                 action_space=4
                 ):
        """
        eps - коеф рандома
        min_eps - минимальный epsilon
        eps_delta - коеф уменьшения epsilonа
        n_jumps - количество прыжков
        action_space - количество доступных действий в среде
        """
        self.EPS = copy.deepcopy(eps)
        self.START_EPS = copy.deepcopy(eps)
        self.MIN_EPS = min_eps
        self.EPS_DELTA = eps_delta
        self.N_JUMPS = n_jumps
        self.action_space = action_space


    def _eps_minimize(self):

        self.EPS = self.EPS*self.EPS_DELTA

        if self.EPS < self.MIN_EPS and self.N_JUMPS > 0:
            self.EPS = copy.deepcopy(self.START_EPS)
            self.N_JUMPS -= 1
        elif self.EPS < self.MIN_EPS:
            self.EPS = copy.deepcopy(self.MIN_EPS)


    def action(self):
        """
        Возвращает None если действие выбирает сеть, иначе номер действия

        """
        rand = np.random.random()
        action = None

        if rand >= 1 - self.EPS:
            action = np.random.choice(range(self.action_space))

        self._eps_minimize()

        return action


