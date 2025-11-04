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

import os
import random
import numpy as np

def set_seed(seed: int) -> None:
    """
    Configura seeds para garantir reprodutibilidade.

    Args:
        seed (int): Valor da seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Nota: Para reprodutibilidade completa também é necessário setar a seed do ambiente
    # ao criá-lo: env.reset(seed=seed)