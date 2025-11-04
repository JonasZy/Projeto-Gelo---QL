# Experimentos e logs do ambiente FrozenLake
Este diretório contém os logs gerados durante o treinamento e teste do agente Q-Learning.

## Estrutura dos arquivos
- `treinamento_[TIMESTAMP].log`: Log detalhado com progresso e métricas
- `q_table_[TIMESTAMP].npy`: Q-table final do experimento
- `results_[TIMESTAMP].json`: Configuração e métricas finais
- `performance_[TIMESTAMP].png`: Gráficos de desempenho

## Interpretação dos resultados
- Taxa de sucesso: % de episódios onde o agente alcançou o objetivo
- Média de passos: Eficiência do agente (menor = melhor)
- Gráficos: Evolução do aprendizado ao longo do tempo