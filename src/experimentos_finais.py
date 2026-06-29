import numpy as np
import pandas as pd
import urllib.request
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import skfuzzy as fuzzy
import skfuzzy.control as ctrl
import warnings

warnings.filterwarnings('ignore')
NUM_EXECUCOES = 21

# =====================================================================
# 1. FUNÇÃO DE DOWNLOAD DIRETO DOS DATASETS DA UCI
# =====================================================================
def carregar_dados_uci():
    print("--- Baixando Datasets Diretamente da UCI Machine Learning Repository ---")
    
    # Dataset 3: Wine Quality (Tinto)
    url_wine = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    req_wine = urllib.request.Request(url_wine, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req_wine) as response:
        df_wine = pd.read_csv(io.StringIO(response.read().decode('utf-8')), sep=';')
    X_wine = df_wine.drop(columns=['quality']).values
    y_wine = df_wine['quality'].values
    
    # Dataset 4: Heart Disease (Statlog)
    url_heart = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
    req_heart = urllib.request.Request(url_heart, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req_heart) as response:
        df_heart = pd.read_csv(io.StringIO(response.read().decode('utf-8')), sep=' ', header=None)
    X_heart = df_heart.iloc[:, :-1].values
    y_heart = df_heart.iloc[:, -1].values - 1 # Mapeia de (1, 2) para (0, 1)
    
    return (X_wine, y_wine), (X_heart, y_heart)

(X_wine, y_wine), (X_heart, y_heart) = carregar_dados_uci()

# =====================================================================
# 2. EXPERIMENTO 3: WINE QUALITY (TAREFA DE REGRESSÃO)
# =====================================================================
print("\n====================================================")
print("👉 INICIANDO EXPERIMENTO 3: WINE QUALITY (REGRESSÃO)")
print("====================================================")

# Seleção de variáveis para o Fuzzy de regressão
selector_wine = SelectKBest(score_func=f_regression, k=2)
X_wine_fuzzy = selector_wine.fit_transform(X_wine, y_wine)

mse_rasa, mae_rasa, r2_rasa = [], [], []
mse_prof, mae_prof, r2_prof = [], [], []
mse_fuz_a, mae_fuz_a, r2_fuz_a = [], [], []

for i in range(NUM_EXECUCOES):
    seed = 42 + i
    
    # Divisão Isonômica para Regressão (sem stratify)
    X_train, X_temp, y_train, y_temp = train_test_split(X_wine, y_wine, test_size=0.40, random_state=seed)
    X_val, X_test, y_test_w, y_test_w = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed)
    
    # Divisão para o Fuzzy usando os 2 melhores atributos
    X_train_f, X_temp_f, _, _ = train_test_split(X_wine_fuzzy, y_wine, test_size=0.40, random_state=seed)
    _, X_test_f, _, _ = train_test_split(X_temp_f, y_temp, test_size=0.50, random_state=seed)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    X_train_fs = scaler.fit_transform(X_train_f)
    X_test_fs = scaler.transform(X_test_f)
    
    # --- RNAs de Regressão ---
    mlp_r = MLPRegressor(hidden_layer_sizes=(16,), activation='relu', max_iter=500, random_state=seed)
    mlp_r.fit(X_train_s, y_train)
    p_rasa = mlp_r.predict(X_test_s)
    mse_rasa.append(mean_squared_error(y_test_w, p_rasa))
    mae_rasa.append(mean_absolute_error(y_test_w, p_rasa))
    r2_rasa.append(r2_score(y_test_w, p_rasa))
    
    mlp_p = MLPRegressor(hidden_layer_sizes=(32, 16), activation='tanh', max_iter=500, random_state=seed)
    mlp_p.fit(X_train_s, y_train)
    p_prof = mlp_p.predict(X_test_s)
    mse_prof.append(mean_squared_error(y_test_w, p_prof))
    mae_prof.append(mean_absolute_error(y_test_w, p_prof))
    r2_prof.append(r2_score(y_test_w, p_prof))
    
    # --- Fuzzy Regressão (Mamdani Adaptado para saída contínua de notas 3 a 8) ---
    v1 = ctrl.Antecedent(np.linspace(-3, 3, 20), 'v1')
    v2 = ctrl.Antecedent(np.linspace(-3, 3, 20), 'v2')
    nota = ctrl.Consequent(np.linspace(3, 8, 20), 'nota')
    
    v1['baixa'] = fuzzy.trimf(v1.universe, [-3, -3, 0])
    v1['alta'] = fuzzy.trimf(v1.universe, [0, 3, 3])
    v2['baixa'] = fuzzy.trimf(v2.universe, [-3, -3, 0])
    v2['alta'] = fuzzy.trimf(v2.universe, [0, 3, 3])
    
    nota['ruim'] = fuzzy.trimf(nota.universe, [3, 3, 5.5])
    nota['bom'] = fuzzy.trimf(nota.universe, [5, 8, 8])
    
    r1 = ctrl.Rule(v1['baixa'] & v2['baixa'], nota['ruim'])
    r2 = ctrl.Rule(v1['alta'] | v2['alta'], nota['bom'])
    
    sim_a = ctrl.ControlSystemSimulation(ctrl.ControlSystem([r1, r2]))
    
    p_fuz = []
    for inst in X_test_fs:
        sim_a.input['v1'] = inst[0]
        sim_a.input['v2'] = inst[1]
        try:
            sim_a.compute()
            p_fuz.append(sim_a.output['nota'])
        except:
            p_fuz.append(5.6) # Média global caso saia do universo
    
    mse_fuz_a.append(mean_squared_error(y_test_w, p_fuz))
    mae_fuz_a.append(mean_absolute_error(y_test_w, p_fuz))
    r2_fuz_a.append(r2_score(y_test_w, p_fuz))

print(f"RNA Rasa     -> MSE Médio: {np.mean(mse_rasa):.4f} | MAE Médio: {np.mean(mae_rasa):.4f} | R²: {np.mean(r2_rasa):.4f}")
print(f"RNA Profunda -> MSE Médio: {np.mean(mse_prof):.4f} | MAE Médio: {np.mean(mae_prof):.4f} | R²: {np.mean(r2_prof):.4f}")
print(f"Fuzzy A      -> MSE Médio: {np.mean(mse_fuz_a):.4f} | MAE Médio: {np.mean(mae_fuz_a):.4f} | R²: {np.mean(r2_fuz_a):.4f}")


# =====================================================================
# 3. EXPERIMENTO 4: HEART DISEASE (TAREFA DE CLASSIFICAÇÃO)
# =====================================================================
print("\n====================================================")
print("👉 INICIANDO EXPERIMENTO 4: HEART DISEASE (CLASSIFICAÇÃO)")
print("====================================================")

selector_heart = SelectKBest(score_func=f_classif, k=2)
X_heart_fuzzy = selector_heart.fit_transform(X_heart, y_heart)

acc_rasa, acc_prof, acc_fuz = [], [], []

for i in range(NUM_EXECUCOES):
    seed = 42 + i
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_heart, y_heart, test_size=0.40, random_state=seed, stratify=y_heart)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)
    
    X_train_f, X_temp_f, _, _ = train_test_split(X_heart_fuzzy, y_heart, test_size=0.40, random_state=seed, stratify=y_heart)
    _, X_test_f, _, _ = train_test_split(X_temp_f, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    X_train_fs = scaler.fit_transform(X_train_f)
    X_test_fs = scaler.transform(X_test_f)
    
    # --- Classificadores Neurais ---
    mlp_c = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', max_iter=500, random_state=seed)
    mlp_c.fit(X_train_s, y_train)
    acc_rasa.append(accuracy_score(y_test, mlp_c.predict(X_test_s)))
    
    mlp_cp = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', max_iter=500, random_state=seed)
    mlp_cp.fit(X_train_s, y_train)
    acc_prof.append(accuracy_score(y_test, mlp_cp.predict(X_test_s)))
    
    # --- Fuzzy Classificação ---
    h1 = ctrl.Antecedent(np.linspace(-3, 3, 20), 'h1')
    h2 = ctrl.Antecedent(np.linspace(-3, 3, 20), 'h2')
    saida = ctrl.Consequent(np.linspace(0, 1, 20), 'saida')
    
    h1['normal'] = fuzzy.trimf(h1.universe, [-3, -3, 0])
    h1['alto'] = fuzzy.trimf(h1.universe, [0, 3, 3])
    h2['normal'] = fuzzy.trimf(h2.universe, [-3, -3, 0])
    h2['alto'] = fuzzy.trimf(h2.universe, [0, 3, 3])
    
    saida['saudavel'] = fuzzy.trimf(saida.universe, [0, 0, 0.6])
    saida['doente'] = fuzzy.trimf(saida.universe, [0.4, 1, 1])
    
    rc1 = ctrl.Rule(h1['normal'] & h2['normal'], saida['saudavel'])
    rc2 = ctrl.Rule(h1['alto'] | h2['alto'], saida['doente'])
    
    sim_h = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rc1, rc2]))
    
    p_heart_f = []
    for inst in X_test_fs:
        sim_h.input['h1'] = inst[0]
        sim_h.input['h2'] = inst[1]
        try:
            sim_h.compute()
            p_heart_f.append(1 if sim_h.output['saida'] >= 0.5 else 0)
        except:
            p_heart_f.append(0)
            
    acc_fuz.append(accuracy_score(y_test, p_heart_f))

print(f"RNA Rasa     -> Acurácia Média: {np.mean(acc_rasa):.4f}  (Desvio: {np.std(acc_rasa):.4f})")
print(f"RNA Profunda -> Acurácia Média: {np.mean(acc_prof):.4f}  (Desvio: {np.std(acc_prof):.4f})")
print(f"Fuzzy Mami   -> Acurácia Média: {np.mean(acc_fuz):.4f}  (Desvio: {np.std(acc_fuz):.4f})")
print("\n--- Experimentos Concluídos com Sucesso! ---")