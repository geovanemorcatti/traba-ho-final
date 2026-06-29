import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris, load_breast_cancer

# =====================================================================
# CONFIGURAÇÃO DOS EXPERIMENTOS (MÍNIMO 21 EXECUÇÕES)
# =====================================================================
NUM_EXECUCOES = 21

# Dicionário para guardar os resultados finais de cada dataset
resultados_globais = {}

print("--- Iniciando a Fase Prática do Trabalho ---")

# Vamos simular o loop para os dois primeiros datasets de classificação
datasets = {
    "Iris": load_iris(return_X_y=True),
    "Breast_Cancer": load_breast_cancer(return_X_y=True)
}

for nome_base, (X, y) in datasets.items():
    print(f"\n>> Processando base de dados: {nome_base}...")
    
    # Listas para guardar a acurácia de cada uma das 21 rodadas
    acuracias_rasa = []
    acuracias_profunda = []
    
    # Loop de repetição exigido para mitigar a aleatoriedade (estocasticidade)
    for i in range(NUM_EXECUCOES):
        # Mudando a semente (seed) a cada rodada para garantir independência
        seed_rodada = 42 + i 
        
        # Divisão isonômica recomendada pelo professor (60% treino / 40% resto)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.40, random_state=seed_rodada, stratify=y
        )
        # Quebrando o resto em 20% Validação e 20% Teste final
        # O erro está no stratify=y
        # Correção: Agora usamos y_temp, que tem o mesmo tamanho de X_temp (60 amostras)
        X_val, X_test, y_val, y_test = train_test_split(
             X_temp, y_temp, test_size=0.50, random_state=seed_rodada, stratify=y_temp
        )
        # Normalização essencial para RNAs
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # --- TREINANDO RNA CONFIGURAÇÃO A (Rasa) ---
        mlp_rasa = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', max_iter=500, random_state=seed_rodada)
        mlp_rasa.fit(X_train_scaled, y_train)
        acuracias_rasa.append(mlp_rasa.score(X_test_scaled, y_test))
        
        # --- TREINANDO RNA CONFIGURAÇÃO B (Profunda) ---
        mlp_profunda = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', max_iter=500, random_state=seed_rodada)
        mlp_profunda.fit(X_train_scaled, y_train)
        acuracias_profunda.append(mlp_profunda.score(X_test_scaled, y_test))

    # =====================================================================
    # CÁLCULO DAS MÉTRICAS (MÉDIA E DESVIO-PADRÃO)
    # =====================================================================
    resultados_globais[nome_base] = {
        "RNA_Rasa_Media": np.mean(acuracias_rasa),
        "RNA_Rasa_Std": np.std(acuracias_rasa),
        "RNA_Profunda_Media": np.mean(acuracias_profunda),
        "RNA_Profunda_Std": np.std(acuracias_profunda)
    }
    
    print(f"Resultado {nome_base} (Rasa): Média = {resultados_globais[nome_base]['RNA_Rasa_Media']:.4f} | Desvio-Padrão = {resultados_globais[nome_base]['RNA_Rasa_Std']:.4f}")
    print(f"Resultado {nome_base} (Profunda): Média = {resultados_globais[nome_base]['RNA_Profunda_Media']:.4f} | Desvio-Padrão = {resultados_globais[nome_base]['RNA_Profunda_Std']:.4f}")

print("\n--- Próximo Passo: Salvar estes números e colocar na tabela do LaTeX ---")