import numpy as np
import pandas as pd
import urllib.request
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from evorbf import RbfClassifier  
import skfuzzy as fuzzy
import skfuzzy.control as ctrl
import warnings

warnings.filterwarnings('ignore')
NUM_EXECUCOES = 21

def carregar_dados_uci():
    print("--- Baixando Datasets Diretamente da UCI Machine Learning Repository ---")
    url_wine = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    req_wine = urllib.request.Request(url_wine, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req_wine) as response:
        df_wine = pd.read_csv(io.StringIO(response.read().decode('utf-8')), sep=';')
    X_wine = df_wine.drop(columns=['quality']).values
    y_wine = df_wine['quality'].values
    
    url_heart = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
    req_heart = urllib.request.Request(url_heart, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req_heart) as response:
        df_heart = pd.read_csv(io.StringIO(response.read().decode('utf-8')), sep=' ', header=None)
    X_heart = df_heart.iloc[:, :-1].values
    y_heart = df_heart.iloc[:, -1].values - 1
    return (X_wine, y_wine), (X_heart, y_heart)

(X_wine, y_wine), (X_heart, y_heart) = carregar_dados_uci()

# =====================================================================
# EXPERIMENTO 3: WINE QUALITY (REGRESSÃO)
# =====================================================================
print("\n====================================================")
print("👉 INICIANDO EXPERIMENTO 3: WINE QUALITY (REGRESSÃO)")
print("====================================================")

selector_wine = SelectKBest(score_func=f_regression, k=2)
X_wine_fuzzy = selector_wine.fit_transform(X_wine, y_wine)

metrics = {m: [] for m in ['mse_rasa', 'mae_rasa', 'r2_rasa', 
                           'mse_prof', 'mae_prof', 'r2_prof', 
                           'mse_rbf', 'mae_rbf', 'r2_rbf',
                           'mse_ts', 'mae_ts', 'r2_ts']}

for i in range(NUM_EXECUCOES):
    seed = 42 + i
    X_train, X_temp, y_train, y_temp = train_test_split(X_wine, y_wine, test_size=0.40, random_state=seed)
    _, X_test, _, y_test_w = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed)
    X_train_f, X_temp_f, _, _ = train_test_split(X_wine_fuzzy, y_wine, test_size=0.40, random_state=seed)
    _, X_test_f, _, _ = train_test_split(X_temp_f, y_temp, test_size=0.50, random_state=seed)
    
    scaler_completo = StandardScaler()
    X_train_s = scaler_completo.fit_transform(X_train)
    X_test_s = scaler_completo.transform(X_test)
    
    scaler_fuzzy = StandardScaler()
    scaler_fuzzy.fit(X_train_f)
    X_test_fs = scaler_fuzzy.transform(X_test_f)
    
    # --- MLP Rasa ---
    mlp_r = MLPRegressor(hidden_layer_sizes=(16,), activation='relu', max_iter=500, random_state=seed)
    mlp_r.fit(X_train_s, y_train)
    p_rasa = mlp_r.predict(X_test_s)
    metrics['mse_rasa'].append(mean_squared_error(y_test_w, p_rasa))
    metrics['mae_rasa'].append(mean_absolute_error(y_test_w, p_rasa))
    metrics['r2_rasa'].append(r2_score(y_test_w, p_rasa))
    
    # --- MLP Profunda ---
    mlp_p = MLPRegressor(hidden_layer_sizes=(32, 16), activation='tanh', max_iter=500, random_state=seed)
    mlp_p.fit(X_train_s, y_train)
    p_prof = mlp_p.predict(X_test_s)
    metrics['mse_prof'].append(mean_squared_error(y_test_w, p_prof))
    metrics['mae_prof'].append(mean_absolute_error(y_test_w, p_prof))
    metrics['r2_prof'].append(r2_score(y_test_w, p_prof))
    
    # --- RBF ---
    rbf_w = RbfClassifier()
    rbf_w.fit(X_train_s, y_train.astype(int))
    p_rbf = rbf_w.predict(X_test_s)
    metrics['mse_rbf'].append(mean_squared_error(y_test_w, p_rbf))
    metrics['mae_rbf'].append(mean_absolute_error(y_test_w, p_rbf))
    metrics['r2_rbf'].append(r2_score(y_test_w, p_rbf))
    
    # --- Fuzzy T-S ---
    p_ts = []
    for inst in X_test_fs:
        mu_baixa = fuzzy.trimf(np.array([inst[0]]), [-3, -3, 0])[0]
        mu_alta = fuzzy.trimf(np.array([inst[0]]), [0, 3, 3])[0]
        regra1_saida = 5.0
        regra2_saida = 7.0
        peso_total = mu_baixa + mu_alta
        y_pred_ts = ((mu_baixa * regra1_saida) + (mu_alta * regra2_saida)) / peso_total if peso_total > 0 else 5.6
        p_ts.append(y_pred_ts)
        
    metrics['mse_ts'].append(mean_squared_error(y_test_w, p_ts))
    metrics['mae_ts'].append(mean_absolute_error(y_test_w, p_ts))
    metrics['r2_ts'].append(r2_score(y_test_w, p_ts))

print(f"MLP Rasa       -> MSE: {np.mean(metrics['mse_rasa']):.4f} | MAE: {np.mean(metrics['mae_rasa']):.4f} | R²: {np.mean(metrics['r2_rasa']):.4f}")
print(f"MLP Profunda   -> MSE: {np.mean(metrics['mse_prof']):.4f} | MAE: {np.mean(metrics['mae_prof']):.4f} | R²: {np.mean(metrics['r2_prof']):.4f}")
print(f"Rede RBF       -> MSE: {np.mean(metrics['mse_rbf']):.4f} | MAE: {np.mean(metrics['mae_rbf']):.4f} | R²: {np.mean(metrics['r2_rbf']):.4f}")
print(f"Fuzzy T-S      -> MSE: {np.mean(metrics['mse_ts']):.4f} | MAE: {np.mean(metrics['mae_ts']):.4f} | R²: {np.mean(metrics['r2_ts']):.4f}")

# =====================================================================
# EXPERIMENTO 4: HEART DISEASE (CLASSIFICAÇÃO)
# =====================================================================
print("\n====================================================")
print("👉 INICIANDO EXPERIMENTO 4: HEART DISEASE (CLASSIFICAÇÃO)")
print("====================================================")

selector_heart = SelectKBest(score_func=f_classif, k=2)
X_heart_fuzzy = selector_heart.fit_transform(X_heart, y_heart)
labels_h = np.unique(y_heart)

def init_c_metrics():
    return {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'matrix': []}

def reg_c_metrics(dic, y_true, y_pred):
    dic['acc'].append(accuracy_score(y_true, y_pred))
    dic['prec'].append(precision_score(y_true, y_pred, average='macro', zero_division=0))
    dic['rec'].append(recall_score(y_true, y_pred, average='macro', zero_division=0))
    dic['f1'].append(f1_score(y_true, y_pred, average='macro', zero_division=0))
    dic['matrix'].append(confusion_matrix(y_true, y_pred, labels=labels_h))

def print_c_report(nome, dic):
    print(f"\n--- {nome} ---")
    print(f"  Acurácia Média: {np.mean(dic['acc']):.4f} (Desvio: {np.std(dic['acc']):.4f})")
    print(f"  Precisão Média:  {np.mean(dic['prec']):.4f}")
    print(f"  Revocação Média: {np.mean(dic['rec']):.4f}")
    print(f"  F1-Score Médio:  {np.mean(dic['f1']):.4f}")
    print("  Matriz de Confusão Acumulada:")
    print(np.sum(dic['matrix'], axis=0))

m_rasa, m_prof, m_rbf, m_fuz = init_c_metrics(), init_c_metrics(), init_c_metrics(), init_c_metrics()

for i in range(NUM_EXECUCOES):
    seed = 42 + i
    X_train, X_temp, y_train, y_temp = train_test_split(X_heart, y_heart, test_size=0.40, random_state=seed, stratify=y_heart)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)
    X_train_f, X_temp_f, _, _ = train_test_split(X_heart_fuzzy, y_heart, test_size=0.40, random_state=seed, stratify=y_heart)
    _, X_test_f, _, _ = train_test_split(X_temp_f, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_fs = scaler.fit_transform(X_train_f)
    X_test_fs = scaler.transform(X_test_f)
    
    # --- MLP Rasa ---
    mlp_c = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', max_iter=500, random_state=seed)
    mlp_c.fit(X_train_s, y_train)
    reg_c_metrics(m_rasa, y_test, mlp_c.predict(X_test_s))
    
    # --- MLP Profunda ---
    mlp_cp = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', max_iter=500, random_state=seed)
    mlp_cp.fit(X_train_s, y_train)
    reg_c_metrics(m_prof, y_test, mlp_cp.predict(X_test_s))
    
    # --- RBF ---
    rbf_h = RbfClassifier()
    rbf_h.fit(X_train_s, y_train)
    reg_c_metrics(m_rbf, y_test, rbf_h.predict(X_test_s))
    
    # --- Fuzzy Mamdani ---
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
        except: p_heart_f.append(0)
            
    reg_c_metrics(m_fuz, y_test, p_heart_f)

print_c_report("MLP Rasa", m_rasa)
print_c_report("MLP Profunda", m_prof)
print_c_report("Rede Neural RBF", m_rbf)
print_c_report("Fuzzy Mamdani", m_fuz)
print("\n--- Experimentos Concluídos com Sucesso! ---")