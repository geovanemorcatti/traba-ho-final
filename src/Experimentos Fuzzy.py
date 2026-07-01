import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier  # <-- MLP de volta para o time!
from evorbf import RbfClassifier  
import skfuzzy as fuzzy
import skfuzzy.control as ctrl
import warnings

warnings.filterwarnings('ignore')

NUM_EXECUCOES = 21

def rodar_experimento_fuzzy(nome_base, X, y):
    print(f"\n>> Processando base: {nome_base}...")
    
    selector = SelectKBest(score_func=f_classif, k=2)
    X_reduzido = selector.fit_transform(X, y)
    
    acc_mlp_rasa = []
    acc_mlp_prof = []
    acc_rbf = []
    acc_fuzzy_a = []
    acc_fuzzy_b = []
    
    for i in range(NUM_EXECUCOES):
        seed_rodada = 42 + i
        
        # Divisão isonômica padrão (60/20/20) para as Redes
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.40, random_state=seed_rodada, stratify=y
        )
        _, X_test, _, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed_rodada, stratify=y_temp
        )
        
        # Partição para os modelos Fuzzy (2 atributos)
        X_train_f, X_temp_f, _, _ = train_test_split(
            X_reduzido, y, test_size=0.40, random_state=seed_rodada, stratify=y
        )
        _, X_test_f, _, _ = train_test_split(
            X_temp_f, y_temp, test_size=0.50, random_state=seed_rodada, stratify=y_temp
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_f_scaled = scaler.fit_transform(X_train_f)
        X_test_f_scaled = scaler.transform(X_test_f)
        
        # --- 1. TREINAMENTO MLP RASA ---
        mlp_rasa = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', max_iter=500, random_state=seed_rodada)
        mlp_rasa.fit(X_train_scaled, y_train)
        acc_mlp_rasa.append(accuracy_score(y_test, mlp_rasa.predict(X_test_scaled)))
        
        # --- 2. TREINAMENTO MLP PROFUNDA ---
        mlp_prof = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', max_iter=500, random_state=seed_rodada)
        mlp_prof.fit(X_train_scaled, y_train)
        acc_mlp_prof.append(accuracy_score(y_test, mlp_prof.predict(X_test_scaled)))
        
        # --- 3. TREINAMENTO DA REDE RBF ---
        try:
            rbf = RbfClassifier()
            rbf.fit(X_train_scaled, y_train)
            acc_rbf.append(accuracy_score(y_test, rbf.predict(X_test_scaled)))
        except:
            # Fallback caso a evorbf exija dados estritamente convertidos
            rbf = RbfClassifier()
            rbf.fit(X_train_scaled, y_train.astype(int))
            acc_rbf.append(accuracy_score(y_test, rbf.predict(X_test_scaled)))
        
        # --- 4. CONFIGURAÇÃO FUZZY MAMDANI A (Triangular) ---
        v1 = ctrl.Antecedent(np.linspace(-3, 3, 50), 'v1')
        v2 = ctrl.Antecedent(np.linspace(-3, 3, 50), 'v2')
        saida = ctrl.Consequent(np.linspace(0, 1, 50), 'saida')
        
        v1['baixa'] = fuzzy.trimf(v1.universe, [-3, -3, 0])
        v1['alta'] = fuzzy.trimf(v1.universe, [0, 3, 3])
        v2['baixa'] = fuzzy.trimf(v2.universe, [-3, -3, 0])
        v2['alta'] = fuzzy.trimf(v2.universe, [0, 3, 3])
        saida['classe_0'] = fuzzy.trimf(saida.universe, [0, 0, 0.6])
        saida['classe_1'] = fuzzy.trimf(saida.universe, [0.4, 1, 1])
        
        r1 = ctrl.Rule(v1['baixa'] & v2['baixa'], saida['classe_0'])
        r2 = ctrl.Rule(v1['alta'] | v2['alta'], saida['classe_1'])
        sim_a = ctrl.ControlSystemSimulation(ctrl.ControlSystem([r1, r2]))
        
        # --- 5. CONFIGURAÇÃO FUZZY MAMDANI B (Gaussiana) ---
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
        sim_b = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rb1, rb2, rb3]))
        
        preds_a, preds_b = [], []
        for inst in X_test_f_scaled:
            sim_a.input['v1'] = inst[0]
            sim_a.input['v2'] = inst[1]
            try:
                sim_a.compute()
                preds_a.append(1 if sim_a.output['saida'] >= 0.5 else 0)
            except: preds_a.append(0)
                
            sim_b.input['v1_g'] = inst[0]
            sim_b.input['v2_g'] = inst[1]
            try:
                sim_b.compute()
                preds_b.append(1 if sim_b.output['saida'] >= 0.5 else 0)
            except: preds_b.append(0)
        
        acc_fuzzy_a.append(accuracy_score(y_test, preds_a))
        acc_fuzzy_b.append(accuracy_score(y_test, preds_b))
        
    print(f"MLP Rasa (Config A)            -> Média: {np.mean(acc_mlp_rasa):.4f} | Desvio: {np.std(acc_mlp_rasa):.4f}")
    print(f"MLP Profunda (Config B)        -> Média: {np.mean(acc_mlp_prof):.4f} | Desvio: {np.std(acc_mlp_prof):.4f}")
    print(f"Rede Neural RBF                -> Média: {np.mean(acc_rbf):.4f} | Desvio: {np.std(acc_rbf):.4f}")
    print(f"Fuzzy Mamdani A (Triangular)   -> Média: {np.mean(acc_fuzzy_a):.4f} | Desvio: {np.std(acc_fuzzy_a):.4f}")
    print(f"Fuzzy Mamdani B (Gaussiana)    -> Média: {np.mean(acc_fuzzy_b):.4f} | Desvio: {np.std(acc_fuzzy_b):.4f}")

X_iris, y_iris = load_iris(return_X_y=True)
rodar_experimento_fuzzy("Iris", X_iris, y_iris)

X_cancer, y_cancer = load_breast_cancer(return_X_y=True)
rodar_experimento_fuzzy("Breast_Cancer", X_cancer, y_cancer)