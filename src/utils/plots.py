"""
Nome do arquivo: plots.py
Data de criação: 04/11/2025
Autor: Jonas da Silva Freitas
Matrícula: 01716338

Descrição:
Funções para geração de gráficos e visualizações

Funcionalidades:
- Plotagem de curvas de aprendizado
- Visualização de resultados de avaliação
- Geração de gráficos comparativos
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

def plot_training_curve(rewards: List[float], filepath: str) -> None:
    """Plota a curva de aprendizado durante o treinamento."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Curva de Aprendizado')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.grid(True)
    
    # Adicionar média móvel
    window = min(100, len(rewards))
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), moving_avg, 'r--', label=f'Média Móvel ({window} episódios)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_evaluation_results(rewards: List[float], stats: Dict, filepath: str) -> None:
    """Plota os resultados da avaliação do agente."""
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Recompensas por episódio
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Recompensa')
    plt.axhline(y=stats['mean'], color='r', linestyle='--', label=f'Média ({stats["mean"]:.2f})')
    plt.fill_between(range(len(rewards)), 
                    np.array(rewards) - stats['std'],
                    np.array(rewards) + stats['std'],
                    alpha=0.2)
    plt.title('Recompensas Durante Avaliação')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Boxplot das recompensas
    plt.subplot(1, 2, 2)
    plt.boxplot(rewards)
    plt.title('Distribuição das Recompensas')
    plt.ylabel('Recompensa Total')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()