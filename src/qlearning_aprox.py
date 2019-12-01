import gym
import numpy as np
import pickle
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from base_agent import BaseAgent

import random

warnings.filterwarnings("ignore", category = ConvergenceWarning)

class MLPQAgent(BaseAgent):
    def __init__(self, env, alpha=0.1, epsilon=0.05, gamma=0.2, possible_actions=4):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.legal_actions = list(range(possible_actions))
        self.networks = {}

        # Valores iniciais
        X = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        y = 0.0

        # Inicia o dicionario
        for i in range(possible_actions):
            self.networks[i] = MLPRegressor(learning_rate_init=self.alpha, hidden_layer_sizes=(100,200))
            self.networks[i].partial_fit([X], [y])

    ## NAO ALTERE OS METODOS save_snapshot e getLegalActions ##
    def save_snapshot(self, name):
        self.snapshots[name] = {x:pickle.dumps(self.networks[x]) for x in self.networkss}

    def getLegalActions(self, state):
        return self.legal_actions


    def getAction(self, state):
        '''
        state: vetor de numeros reais

        retorna um valor presente na lista self.getLegalActions(state)
        ''' 
        actions = self.getLegalActions(state)
        results = []
        # Escolhe um valor aleatorio entre 0 e 1
        # Caso o valor for menor que Epsilon, escolhe uma acao aleatoria
        # Caso contrario, escolhe a melhor acao
        if self.epsilon > random.random():
            action = random.choice(actions)
        else:
            for network in self.networks.values():
                X = state.reshape(1, -1)
                results.append(network.predict(X))
            action = results.index(max(results))
        return action


    def update(self, state, action, nextState, reward):
        '''
        Atualiza a rede na posicao self.networks[action].
        '''
        results = []

        for network in self.networks.values():
            X = nextState.reshape(1, -1)
            results.append(network.predict(X))
        max_q = max(results)

        self.networks[action].partial_fit([state], max_q * self.gamma + reward)
