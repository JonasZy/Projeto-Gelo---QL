"""
Nome do arquivo: eval.py
Data de criação: 04/11/2025
Autor: Jonas da Silva Freitas
Matrícula: 01716338

Descrição:
Script para avaliação dos agentes treinados e geração de gráficos

Funcionalidades:
- Carregamento de agentes treinados
- Avaliação de performance
- Geração de gráficos e métricas
- Exportação de resultados
"""

import os
import argparse
import json
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from gymnasium.spaces import Discrete

from src.agents.qlearning import QLearningAgent
from src.agents.sarsa import SarsaAgent
from src.utils.metrics import load_metrics, calculate_stats
from src.utils.plots import plot_evaluation_results

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RL agent')
    parser.add_argument('--run', type=str, required=True, help='Path to training run directory')
    parser.add_argument('--episodes', type=int, default=200, help='Number of evaluation episodes')
    parser.add_argument('--render', type=bool, default=False, help='Render environment')
    parser.add_argument('--export', type=str, required=True, help='Export directory for results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Carregar configuração do treinamento
    with open(os.path.join(args.run, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Inicializar ambiente
    env = gym.make(config['env'], render_mode='human' if args.render else None)
    
    # Carregar agente
    if config['algo'] == 'qlearning':
        agent = QLearningAgent.load(os.path.join(args.run, 'model.pkl'))
    else:
        agent = SarsaAgent.load(os.path.join(args.run, 'model.pkl'))
    
    # Avaliação
    rewards = []
    for episode in range(args.episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state, eval=True)  # Sem exploração durante avaliação
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            
        rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{args.episodes}, Reward: {total_reward:.2f}")
    
    # Calcular estatísticas
    stats = calculate_stats(rewards)
    
    # Criar diretório de exportação
    os.makedirs(args.export, exist_ok=True)
    
    # Salvar resultados
    results = {
        'config': config,
        'evaluation': {
            'episodes': args.episodes,
            'stats': stats,
            'rewards': rewards
        }
    }
    
    with open(os.path.join(args.export, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Gerar gráficos
    plot_evaluation_results(rewards, stats, os.path.join(args.export, 'evaluation_plot.png'))

if __name__ == "__main__":
    main()