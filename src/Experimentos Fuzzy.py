import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score
import skfuzzy as fuzzy
import skfuzzy.control as ctrl
import warnings

warnings.filterwarnings('ignore') # Oculta avisos de indefinição matemática do skfuzzy

NUM_EXECUCOES = 21

def rodar_experimento_fuzzy(nome_base, X, y):
    print(f"\n>> Processando base Fuzzy: {nome_base}...")
    
    # 1. Selecionar os 2 melhores atributos para tornar o Sistema Fuzzy viável computacionalmente
    selector = SelectKBest(score_func=f_classif, k=2)
    X_reduzido = selector.fit_transform(X, y)
    
    acuracias_config_a = []
    acuracias_config_b = []
    
    for i in range(NUM_EXECUCOES):
        seed_rodada = 42 + i
        
        # Divisão isonômica identica ao teste anterior (60/20/20)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_reduzido, y, test_size=0.40, random_state=seed_rodada, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed_rodada, stratify=y_temp
        )
        
        # Normalização dos limites para facilitar o universo Fuzzy (0 a 10)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Mapeamento do Universo de Discurso (Escala Padronizada de -3 a 3)
        v1 = ctrl.Antecedent(np.linspace(-3, 3, 50), 'v1')
        v2 = ctrl.Antecedent(np.linspace(-3, 3, 50), 'v2')
        saida = ctrl.Consequent(np.linspace(0, 1, 50), 'saida')
        
        # =================================================================
        # CONFIGURAÇÃO FUZZY A: Funções Triangulares (Poucas Regras)
        # =================================================================
        v1['baixa'] = fuzzy.trimf(v1.universe, [-3, -3, 0])
        v1['alta'] = fuzzy.trimf(v1.universe, [0, 3, 3])
        
        v2['baixa'] = fuzzy.trimf(v2.universe, [-3, -3, 0])
        v2['alta'] = fuzzy.trimf(v2.universe, [0, 3, 3])
        
        saida['classe_0'] = fuzzy.trimf(saida.universe, [0, 0, 0.6])
        saida['classe_1'] = fuzzy.trimf(saida.universe, [0.4, 1, 1])
        
        r1 = ctrl.Rule(v1['baixa'] & v2['baixa'], saida['classe_0'])
        r2 = ctrl.Rule(v1['alta'] | v2['alta'], saida['classe_1'])
        
        sis_a = ctrl.ControlSystem([r1, r2])
        sim_a = ctrl.ControlSystemSimulation(sis_a)
        
        # =================================================================
        # CONFIGURAÇÃO FUZZY B: Funções Gaussianas (Mais Regras)
        # =================================================================
        v1_g = ctrl.Antecedent(np.linspace(-3, 3, 50), 'v1_g')
        v2_g = ctrl.Antecedent(np.linspace(-3, 3, 50), 'v2_g')
        
        v1_g['baixa'] = fuzzy.gaussmf(v1_g.universe, -1.5, 0.8)
        v1_g['media'] = fuzzy.gaussmf(v1_g.universe, 0.0, 0.8)
        v1_g['alta'] = fuzzy.gaussmf(v1_g.universe, 1.5, 0.8)
        
        v2_g['baixa'] = fuzzy.gaussmf(v2_g.universe, -1.5, 0.8)
        v2_g['alta'] = fuzzy.gaussmf(v2_g.universe, 1.5, 0.8)
        
        rb1 = ctrl.Rule(v1_g['baixa'] & v2_g['baixa'], saida['classe_0'])
        rb2 = ctrl.Rule(v1_g['media'], saida['classe_0'])
        rb3 = ctrl.Rule(v1_g['alta'] | v2_g['alta'], saida['classe_1'])
        
        sis_b = ctrl.ControlSystem([rb1, rb2, rb3])
        sim_b = ctrl.ControlSystemSimulation(sis_b)
        
        # --- Inferência no Conjunto de Teste ---
        preds_a = []
        preds_b = []
        
        for inst in X_test_scaled:
            # Teste Configuração A
            sim_a.input['v1'] = inst[0]
            sim_a.input['v2'] = inst[1]
            try:
                sim_a.compute()
                preds_a.append(1 if sim_a.output['saida'] >= 0.5 else 0)
            except:
                preds_a.append(0) # Fallback se cair fora das regras
                
            # Teste Configuração B
            sim_b.input['v1_g'] = inst[0]
            sim_b.input['v2_g'] = inst[1]
            try:
                sim_b.compute()
                preds_b.append(1 if sim_b.output['saida'] >= 0.5 else 0)
            except:
                preds_b.append(0)
        
        acuracias_config_a.append(accuracy_score(y_test, preds_a))
        acuracias_config_b.append(accuracy_score(y_test, preds_b))
        
    print(f"Resultado {nome_base} (Fuzzy A - Triangular): Média = {np.mean(acuracias_config_a):.4f} | Desvio-Padrão = {np.std(acuracias_config_a):.4f}")
    print(f"Resultado {nome_base} (Fuzzy B - Gaussiana): Média = {np.mean(acuracias_config_b):.4f} | Desvio-Padrão = {np.std(acuracias_config_b):.4f}")

# Executar para as duas bases
X_iris, y_iris = load_iris(return_X_y=True)
rodar_experimento_fuzzy("Iris", X_iris, y_iris)

X_cancer, y_cancer = load_breast_cancer(return_X_y=True)
rodar_experimento_fuzzy("Breast_Cancer", X_cancer, y_cancer)