"""
Nome do arquivo: qlearning.py
Data de criação: 04/11/2025
Autor: Jonas da Silva Freitas
Matrícula: 01716338

Descrição:
Implementação do algoritmo Q-Learning

Funcionalidades:
- Inicialização da Q-table
- Seleção de ações (epsilon-greedy)
- Atualização dos valores Q
- Salvamento e carregamento do modelo
"""

import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Taxa de aprendizado
        self.gamma = gamma  # Fator de desconto
        self.epsilon = epsilon_start  # Taxa de exploração
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Inicializar Q-table
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state, eval=False):
        if not eval and np.random.random() < self.epsilon:
            # Exploração: escolher ação aleatória
            return np.random.choice(self.action_size)
        else:
            # Exploitation: escolher melhor ação
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        # Atualizar valor Q usando a equação do Q-Learning
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
        # Decair epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'state_size': self.state_size,
                'action_size': self.action_size,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay
            }, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        agent = cls(
            state_size=data['state_size'],
            action_size=data['action_size'],
            alpha=data['alpha'],
            gamma=data['gamma'],
            epsilon_start=data['epsilon'],
            epsilon_min=data['epsilon_min'],
            epsilon_decay=data['epsilon_decay']
        )
        agent.q_table = data['q_table']
        return agent