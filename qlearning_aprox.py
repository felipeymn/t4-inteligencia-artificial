import gym
import numpy as np
import pickle
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from base_agent import BaseAgent

warnings.filterwarnings("ignore", category = ConvergenceWarning)

class MLPQAgent(BaseAgent):
    def __init__(self, env, alpha=0.1, epsilon=0.05, gamma=0.2, possible_actions=4):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.legal_actions = list(range(possible_actions))

        # Utilize o dicionario networks de forma que cada indice eh representado
        # por um valor entre 0 e possible_actions-1
        # O dicionario devera guardar as instancias da classe MLPRegressor
        # LEMBRE DE INSTANCIAR AS REDES EM ALGUM LUGAR
        self.networks = {}


    ## NAO ALTERE OS METODOS save_snapshot e getLegalActions ##
    def save_snapshot(self, name):
        self.snapshots[name] = {x:pickle.dumps(self.networks[x]) for x in self.networks}

    def getLegalActions(self, state):
        return self.legal_actions


    def getAction(self, state):
        '''
        state: vetor de numeros reais

        retorna um valor presente na lista self.getLegalActions(state)
        '''
        # IMPLEMENTE AQUI O METODO PARA ESCOLHER A ACAO
        actions = self.getLegalActions(state)
        i = np.random.randint(len(actions))
        return actions[i]

    def update(self, state, action, nextState, reward):
        '''
        Atualiza a rede na posicao self.networks[action].
        '''
        # IMPLEMENTE AQUI O METODO PARA ATUALIZAR O AGENTE
        pass