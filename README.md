# Projeto GELO - Experimento Q-Learning com FrozenLake 🧊

## Introdução
Este projeto implementa um agente de aprendizado por reforço (Reinforcement Learning) usando o algoritmo Q-Learning para resolver o ambiente FrozenLake-v1 do Gymnasium (fork moderno do OpenAI Gym). O agente aprende a navegar em um lago congelado, evitando buracos, para alcançar o objetivo.

## Tecnologias e Bibliotecas
- **Python 3.x**
- **Principais dependências:**
  - `gymnasium`: Ambiente de simulação (FrozenLake-v1)
  - `numpy`: Operações numéricas e Q-table
  - `matplotlib`: Visualização de resultados

## Algoritmos e Conceitos Aplicados

### Q-Learning
Q-Learning é um algoritmo de aprendizado por reforço off-policy que aprende uma função de valor-ação (Q-function) através de experiências. 

#### Componentes Principais:
1. **Q-Table**: Matriz que armazena valores Q(s,a) para cada par estado-ação
2. **Política ε-greedy**: Balanço entre exploração e aproveitamento
3. **Atualização de Bellman**: Fórmula central do Q-Learning

### Fórmulas e Cálculos Principais

#### 1. Atualização Q-Learning (Equação de Bellman):

Q(s,a) ← Q(s,a) + α[r + γ * max_a'(Q(s',a')) - Q(s,a)]

Onde:
- α (taxa_aprendizagem): Taxa de aprendizado (0.1)
- γ (fator_Desconto): Fator de desconto para recompensas futuras (0.99)
- r: Recompensa imediata
- s: Estado atual
- a: Ação tomada
- s': Próximo estado
- max_a'(Q(s',a')): Máximo valor Q possível no próximo estado

#### 2. Decaimento do ε (Epsilon):

ε = ε_min + (ε_max - ε_min) * exp(-decay_rate * episode)

- Controla o balanço exploração/aproveitamento
- Começa com mais exploração (ε=1.0) e gradualmente aumenta aproveitamento

## Como Executar

### Instalação
bash
# Clone o repositório
git clone [URL_DO_REPO]
cd projeto-gelo

# Crie e ative um ambiente virtual (recomendado)
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# Instale as dependências
pip install -r requirements.txt


### Execução

#### Treino Básico
bash
python src/gelo.py


#### Opções de Configuração
bash
# Treino completo com visualização
python src/gelo.py --total-episodes 5000 --num-test-episodes 500 --visualize-test

# Treino rápido para testes
python src/gelo.py --total-episodes 1000 --num-test-episodes 100 --render-delay 0.005

# Ajuda com todos os parâmetros
python src/gelo.py --help


### Parâmetros Principais
- `--total-episodes`: Número de episódios de treino
- `--num-test-episodes`: Número de episódios de teste
- `--visualize-test`: Ativa visualização dos testes
- `--render-delay`: Controla velocidade da animação
- `--plot-results`: Gera gráficos de desempenho
- `--log-dir`: Diretório para salvar logs/resultados

## Resultados e Análise

### Arquivos Gerados
O experimento gera vários arquivos no diretório `logs/`:
- `treinamento_[TIMESTAMP].log`: Log detalhado do treino
- `q_table_[TIMESTAMP].npy`: Q-table final salva
- `results_[TIMESTAMP].json`: Métricas e configuração
- `performance_[TIMESTAMP].png`: Gráficos de desempenho

### Métricas Coletadas
- Taxa de sucesso (% episódios completados)
- Média de passos por episódio
- Recompensa média
- Evolução do aprendizado (via gráficos)

### Visualizações
O script gera dois gráficos principais:
1. **Passos por Episódio**: Mostra a eficiência do agente
2. **Recompensa por Episódio**: Indica o sucesso do aprendizado

## Comentários Finais

### Pontos Fortes
- Implementação completa de Q-Learning
- Sistema robusto de logs e métricas
- Visualizações claras do progresso
- Flexibilidade via argumentos de linha de comando

### Limitações Conhecidas
- Ambiente discreto apenas (FrozenLake)
- Renderização pode não funcionar em ambientes headless
- Q-table pode ser grande para estados/ações numerosos

### Próximos Passos Possíveis
- Implementar outros algoritmos (DQN)
- Melhorar visualizações em tempo real
- Comparar diferentes hiperparâmetros

## Autor
Projeto-Gelo
Jonas da Silva Freitas
Matrícula: 01716338