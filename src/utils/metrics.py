"""
Nome do arquivo: metrics.py
Data de criação: 04/11/2025
Autor: Jonas da Silva Freitas
Matrícula: 01716338

Descrição:
Funções para cálculo e logging de métricas de treinamento

Funcionalidades:
- Logging de métricas
- Cálculo de estatísticas
- Carregamento de métricas salvas
"""

import json
import numpy as np
from typing import List, Dict

def log_metrics(rewards: List[float], filepath: str) -> None:
    """Salva métricas de treinamento em arquivo JSON."""
    metrics = {
        'rewards': rewards,
        'stats': calculate_stats(rewards)
    }
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(filepath: str) -> Dict:
    """Carrega métricas de um arquivo JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_stats(rewards: List[float]) -> Dict:
    """Calcula estatísticas básicas sobre as recompensas."""
    return {
        'mean': float(np.mean(rewards)),
        'std': float(np.std(rewards)),
        'min': float(np.min(rewards)),
        'max': float(np.max(rewards)),
        'median': float(np.median(rewards))
    }