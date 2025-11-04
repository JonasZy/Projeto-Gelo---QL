"""
Nome do arquivo: train.py
Data de criação: 04/11/2025
Autor: Jonas da Silva Freitas
Matrícula: 01716338

Descrição:
Script principal para treinamento dos agentes de aprendizado por reforço

Funcionalidades:
- Configuração e inicialização do ambiente
- Treinamento de agentes (Q-Learning/SARSA)
- Logging de métricas de treinamento
- Salvamento de checkpoints do modelo
"""

import os
import argparse
import json
import gymnasium as gym
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.qlearning import QLearningAgent
from src.agents.sarsa import SarsaAgent
from src.utils.metrics import log_metrics
from src.utils.plots import plot_training_curve
from src.utils.seed import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent')
    parser.add_argument('--algo', type=str, choices=['qlearning', 'sarsa'], required=True)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_min', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Criar diretório de saída
    os.makedirs(args.out, exist_ok=True)
    
    # Salvar configuração
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Inicializar ambiente
    env = gym.make(args.env)
    
    # Inicializar agente
    if args.algo == 'qlearning':
        agent = QLearningAgent(
            state_size=env.observation_space.n,
            action_size=env.action_space.n,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay
        )
    else:
        agent = SarsaAgent(
            state_size=env.observation_space.n,
            action_size=env.action_space.n,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay
        )
    
    # Loop de treinamento
    rewards_history = []
    for episode in range(args.episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{args.episodes}, Avg Reward: {np.mean(rewards_history[-100:]):.2f}")
    
    # Salvar resultados
    log_metrics(rewards_history, os.path.join(args.out, 'metrics.json'))
    plot_training_curve(rewards_history, os.path.join(args.out, 'learning_curve.png'))
    agent.save(os.path.join(args.out, 'model.pkl'))

if __name__ == "__main__":
    main()