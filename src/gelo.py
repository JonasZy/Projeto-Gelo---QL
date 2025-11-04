import gymnasium as gym # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import time
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
import os

# Configuração de argumentos da linha de comando
parser = argparse.ArgumentParser(description='Experimento Q-Learning com FrozenLake')
parser.add_argument('--total-episodes', type=int, default=5000, help='Número total de episódios de treino')
parser.add_argument('--num-test-episodes', type=int, default=500, help='Número de episódios de teste')
parser.add_argument('--visualize-test', action='store_true', help='Visualizar os testes')
parser.add_argument('--plot-results', action='store_true', default=True, help='Gerar gráfico de resultados')
parser.add_argument('--render-delay', type=float, default=0.01, help='Delay entre frames (segundos)')
parser.add_argument('--log-dir', type=str, default='logs', help='Diretório para logs e resultados')

args = parser.parse_args()

# Criar diretório de logs se não existir
os.makedirs(args.log_dir, exist_ok=True)
experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(args.log_dir, f'treinamento_{experiment_time}.log')

# 1. Configuração
# --------------------------------------------------------------------------
# Escolha o ambiente. Exemplo: FrozenLake
env = gym.make("FrozenLake-v1", render_mode="human",)# is_slippery=False para começar mais fácil

# Obter o tamanho do espaço de estados e ações
n_states = env.observation_space.n   # Ex: 16 para FrozenLake 4x4
n_actions = env.action_space.n       # Ex: 4 para FrozenLake (Cima, Baixo, Esquerda, Direita)

# Inicializar a Q-TabelaQ_tabela (todos os Q-valores em zero ou pequenos números aleatórios)
Q_tabela = np.zeros((n_states, n_actions))

# Hyperparâmetros do Q-Learning
taxa_aprendizagem = 0.1  # Taxa de aprendizado (alpha)
fator_Disconto = 0.99 # Fator de desconto (gamma)
base_ = 1.0        # Taxa de exploração inicial
maximo_epsilon = 1.0
minimo_epsilon = 0.01
taxa_decaimento = 0.0001  # Taxa de decaimento do base_

# 2. Loop de Treinamento
# --------------------------------------------------------------------------
def log_info(message):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

log_info(f"\nIniciando treinamento com {args.total_episodes} episódios")
log_info(f"Configuração:\n{json.dumps(vars(args), indent=2, ensure_ascii=False)}\n")

# Métricas de treinamento
rewards_history = []
steps_history = []

for episode in range(args.total_episodes):
    # Iniciar um novo episódio
    estado, info = env.reset()
    terminado = False
    truncado = False
    
    while not terminado and not truncado:
        
        # 2.1 Escolher a Ação (Epsilon-Greedy)
        # ------------------------------------
        if np.random.random() < base_:
            # Exploração: Escolhe uma ação aleatória
            acao = env.action_space.sample() 
        else:
            # Aproveitamento: Escolhe a ação com o maior Q-valor (Melhor política)
            acao = np.argmax(Q_tabela[estado,:])

        # 2.2 Executar a Ação
        # --------------------
        novo_estado, recompensa, terminado, truncado, info = env.step(acao)
        
        # Calcular o Valor Q Antigo
        antigo_valor = Q_tabela[estado, acao]
        
        # Calcular o Valor Q Máximo Futuro (max a' Q(s', a'))
        future_max_q = np.max(Q_tabela[novo_estado,:])
        
        # Aplicar a Equação de Bellman para Q-Learning:
        # Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a'(Q(s', a')) - Q(s, a) ]
        valor_novo = antigo_valor + taxa_aprendizagem * (
            recompensa + fator_Disconto * future_max_q - antigo_valor
        )
        
        # Armazenar o novo valor
        Q_tabela[estado, acao] = valor_novo
        
        # Mover para o próximo estado
        estado = novo_estado
        
    
    base_ = minimo_epsilon + (maximo_epsilon - minimo_epsilon) * np.exp(-taxa_decaimento * episode)


# 3. Teste (Opcional, mas recomendado)
# --------------------------------------------------------------------------
print("Treinamento concluído. Q-Tabela final:")
# print(Q_tabela) # Pode ser muito grande

# Parâmetros de visualização/teste
visualize_test = False  # Se True, renderiza cada passo durante os testes (pode ser lento)
plot_results = True     # Se True, gera um gráfico (Episódio x Número de passos)

# Testa o agente treinado por N episódios e registra o número de passos até terminar
test_steps = []
num_test_episodes = 500
for test_episode in range(num_test_episodes):
    estado, info = env.reset()
    terminado = False
    truncado = False
    passo_count = 0
    # print("--- Novo Teste ---")
    
    episode_reward = 0
    while not terminado and not truncado:
        # Ação é sempre a melhor (sem exploração: base_ = 0)
        acao = np.argmax(Q_tabela[estado,:]) 
        
        novo_estado, recompensa, terminado, truncado, info = env.step(acao)
        episode_reward += recompensa
        passo_count += 1

        # Renderiza opcionalmente para ver o agente
        if args.visualize_test:
            try:
                env.render()
            except Exception:
                pass
            time.sleep(args.render_delay)

        # Verifica término e imprime resultado simples
        if terminado:
            if recompensa > 0:
                print(f"Teste {test_episode+1}: Sucesso! Recompensa: {recompensa} (passos: {passo_count})")
            else:
                print(f"Teste {test_episode+1}: Falha. Recompensa: {recompensa} (passos: {passo_count})")

        estado = novo_estado

    test_steps.append(passo_count)
    rewards_history.append(episode_reward)
    steps_history.append(passo_count)

    if (test_episode + 1) % 50 == 0:
        log_info(f"Teste {test_episode+1}/{args.num_test_episodes} completado")
        log_info(f"Média de passos (últimos 50): {np.mean(test_steps[-50:]):.2f}")
        log_info(f"Média de recompensas (últimos 50): {np.mean(rewards_history[-50:]):.2f}\n")

# Salvar Q-tabela e métricas
results = {
    'config': vars(args),
    'metrics': {
        'total_steps': sum(test_steps),
        'mean_steps': float(np.mean(test_steps)),
        'mean_reward': float(np.mean(rewards_history)),
        'success_rate': float(sum(r > 0 for r in rewards_history) / len(rewards_history))
    }
}

# Salvar resultados
np.save(os.path.join(args.log_dir, f'q_table_{experiment_time}.npy'), Q_tabela)
with open(os.path.join(args.log_dir, f'results_{experiment_time}.json'), 'w') as f:
    json.dump(results, f, indent=2)

log_info("\nResultados finais:")
log_info(f"Taxa de sucesso: {results['metrics']['success_rate']*100:.1f}%")
log_info(f"Média de passos por episódio: {results['metrics']['mean_steps']:.2f}")
log_info(f"Q-table e resultados salvos em {args.log_dir}/")

# Fecha o ambiente ao final
env.close()

# Gera o gráfico Episódio x Número de passos
if args.plot_results:
    try:
        x = list(range(1, len(test_steps) + 1))
        y = test_steps
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker='o', linestyle='-', markersize=3)
        plt.xlabel('Episódio')
        plt.ylabel('Número de passos')
        plt.title('Desempenho do agente: Episódio vs Número de passos')
        plt.grid(True)
        plt.tight_layout()
        # Criar subplots para steps e rewards
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot de steps
        ax1.plot(x, y, marker='o', linestyle='-', markersize=3)
        ax1.set_xlabel('Episódio')
        ax1.set_ylabel('Número de passos')
        ax1.set_title('Desempenho do agente: Passos por Episódio')
        ax1.grid(True)
        
        # Plot de rewards
        ax2.plot(x, rewards_history, marker='o', linestyle='-', markersize=3, color='green')
        ax2.set_xlabel('Episódio')
        ax2.set_ylabel('Recompensa')
        ax2.set_title('Desempenho do agente: Recompensa por Episódio')
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(args.log_dir, f'performance_{experiment_time}.png')
        plt.savefig(plot_path)
        try:
            plt.show()
        except Exception:
            pass
        log_info(f'Gráficos salvos em {plot_path}')
    except Exception as e:
        print('Erro ao gerar gráfico:', e)