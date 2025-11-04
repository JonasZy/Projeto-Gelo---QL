"""
Nome do arquivo: seed.py
Data de criação: 04/11/2025
Autor: Jonas da Silva Freitas
Matrícula: 01716338

Descrição:
Funções para garantir reprodutibilidade dos experimentos

Funcionalidades:
- Configuração de seeds aleatórias
- Garantia de reprodutibilidade
"""

import random
import numpy as np
import gymnasium as gym

def set_seed(seed: int) -> None:
    """
    Configura seeds para garantir reprodutibilidade.
    
    Args:
        seed (int): Valor da seed
    """
    random.seed(seed)
    np.random.seed(seed)
    gym.utils.seeding.create_seed(seed)