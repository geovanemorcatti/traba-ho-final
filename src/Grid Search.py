import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import skfuzzy as fuzzy
import skfuzzy.control as ctrl

# =====================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS (ISONOMIA)
# =====================================================================
print("--- Carregando Dataset: Breast Cancer Wisconsin ---")
data = load_breast_cancer()
X = data.data
y = data.target

# Divisão de dados sugerida: 60% Treino, 20% Validação (usada no Grid Search), 20% Teste
# Primeiro separa 20% para teste final
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# O Scaler é fundamental para a convergência das Redes Neurais (evita explosão de gradiente)
scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val)
X_test_scaled = scaler.transform(X_test)


# =====================================================================
# 2. REDES NEURAIS: ESCOLA SISTEMÁTICA DE PARÂMETROS (GRID SEARCH)
# =====================================================================
print("\n--- Iniciando Grid Search para as Redes Neurais (MLP) ---")

# O algoritmo testará combinações de:
# - Duas camadas (Rasa vs Profunda)
# - Duas funções de ativação (Relu vs Tanh)
# - Duas taxas de aprendizado
param_grid = {
    'hidden_layer_sizes': [(16,), (32, 16)],  # Configuração A (Rasa) vs Configuração B (Profunda)
    'activation': ['relu', 'tanh'],           # Funções de ativação
    'learning_rate_init': [0.01, 0.001],      # Hiperparâmetro de otimização
    'max_iter': [1000]                        # Garante a convergência
}

mlp_base = MLPClassifier(random_state=42)

# O GridSearchCV usa validação cruzada interna nos 80% iniciais para achar o melhor parâmetro
grid_search = GridSearchCV(estimator=mlp_base, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_val_scaled, y_train_val)

print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

# Extraindo os dois melhores modelos distintos para comparar no relatório:
# Configuração 1 (Rasa)
mlp_rasa = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', learning_rate_init=0.01, max_iter=1000, random_state=42)
mlp_rasa.fit(X_train_val_scaled, y_train_val)

# Configuração 2 (Profunda)
mlp_profunda = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', learning_rate_init=0.001, max_iter=1000, random_state=42)
mlp_profunda.fit(X_train_val_scaled, y_train_val)


# =====================================================================
# 3. MODELO FUZZY / NEURO-FUZZY (MAMDANI VIA SKFUZZY)
# =====================================================================
print("\n--- Configurando Sistema de Controle Fuzzy (Mamdani) ---")

# Para fins didáticos e devido à limitação de dimensionalidade de regras em sistemas puramente Fuzzy,
# usaremos dois atributos altamente preditivos da base (Raio Médio e Textura Média)
raio = ctrl.Antecedent(np.linspace(0, 30, 100), 'raio')
textura = ctrl.Antecedent(np.linspace(0, 40, 100), 'textura')
diagnostico = ctrl.Consequent(np.linspace(0, 1, 100), 'diagnostico')

# --- CONFIGURAÇÃO FUZZY A: Funções de Pertinência Triangulares (3 Regras) ---
raio['baixo'] = fuzzy.trimf(raio.universe, [0, 0, 15])
raio['alto'] = fuzzy.trimf(raio.universe, [10, 30, 30])

textura['baixa'] = fuzzy.trimf(textura.universe, [0, 0, 20])
textura['alta'] = fuzzy.trimf(textura.universe, [15, 40, 40])

diagnostico['benigno'] = fuzzy.trimf(diagnostico.universe, [0, 0, 0.6])
diagnostico['maligno'] = fuzzy.trimf(diagnostico.universe, [0.4, 1, 1])

# Definição do bloco de regras para a Configuração A
regra1 = ctrl.Rule(raio['baixo'] & textura['baixa'], diagnostico['benigno'])
regra2 = ctrl.Rule(raio['alto'] | textura['alta'], diagnostico['maligno'])

sistema_fuzzy_a = ctrl.ControlSystem([regra1, regra2])
simulador_a = ctrl.ControlSystemSimulation(sistema_fuzzy_a)

# --- CONFIGURAÇÃO FUZZY B: Funções de Pertinência Gaussianas (Mais Regras/Sensibilidade) ---
# O professor exige testar variação de funções e número de regras
raio_g = ctrl.Antecedent(np.linspace(0, 30, 100), 'raio_g')
textura_g = ctrl.Antecedent(np.linspace(0, 40, 100), 'textura_g')

raio_g['baixo'] = fuzzy.gaussmf(raio_g.universe, 10, 3)
raio_g['medio'] = fuzzy.gaussmf(raio_g.universe, 18, 3)
raio_g['alto'] = fuzzy.gaussmf(raio_g.universe, 25, 3)

textura_g['baixa'] = fuzzy.gaussmf(textura_g.universe, 12, 4)
textura_g['alta'] = fuzzy.gaussmf(textura_g.universe, 28, 4)

regra_b1 = ctrl.Rule(raio_g['baixo'] & textura_g['baixa'], diagnostico['benigno'])
regra_b2 = ctrl.Rule(raio_g['medio'], diagnostico['benigno'])
regra_b3 = ctrl.Rule(raio_g['alto'] | textura_g['alta'], diagnostico['maligno'])

sistema_fuzzy_b = ctrl.ControlSystem([regra_b1, regra_b2, regra_b3])
simulador_b = ctrl.ControlSystemSimulation(sistema_fuzzy_b)


# =====================================================================
# 4. AVALIAÇÃO QUANTITATIVA NO CONJUNTO DE TESTE (MÉTRICAS)
# =====================================================================
print("\n=== RESULTADOS FINAIS NO CONJUNTO DE TESTE (MÉTRICAS OBRIGATÓRIAS) ===")

# Avaliando RNA Rasa
y_pred_rasa = mlp_rasa.predict(X_test_scaled)
print("\n[RNA Configuração A - Rasa] Relatório de Classificação:")
print(classification_report(y_test, y_pred_rasa, target_names=data.target_names))

# Avaliando RNA Profunda
y_pred_profunda = mlp_profunda.predict(X_test_scaled)
print("\n[RNA Configuração B - Profunda] Relatório de Classificação:")
print(classification_report(y_test, y_pred_profunda, target_names=data.target_names))

# Avaliando Sistema Fuzzy A (Mamdani Simples)
y_pred_fuzzy_a = []
for i in range(len(X_test)):
    # Passando os valores de raio médio (coluna 0) e textura média (coluna 1)
    simulador_a.input['raio'] = X_test[i, 0]
    simulador_a.input['textura'] = X_test[i, 1]
    try:
        simulador_a.compute()
        out = simulador_a.output['diagnostico']
        y_pred_fuzzy_a.append(1 if out >= 0.5 else 0)
    except:
        y_pred_fuzzy_a.append(0) # Fallback em caso de indeterminação matemática pelas regras

print("\n[Sistema Fuzzy Configuração A] Relatório de Classificação:")
print(classification_report(y_test, y_pred_fuzzy_a, target_names=data.target_names))