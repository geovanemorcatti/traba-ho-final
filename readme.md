# Comparação de Desempenho: Redes Neurais Artificiais vs. Sistemas Fuzzy

Este projeto foi desenvolvido como requisito para a atividade prática da disciplina de **Inteligência Computacional** do **CEFET-MG (Semestre 01/2026)**, ministrada pelo **Prof. [cite_start]Alisson Marques da Silva**[cite: 1, 2, 30].

[cite_start]O objetivo do trabalho é avaliar de forma justa, rigorosa e isonômica duas configurações de Redes Neurais Artificiais (MLP) e duas abordagens baseadas em Lógica Fuzzy/Neuro-Fuzzy em 4 conjuntos de dados reais do repositório *UCI Machine Learning*[cite: 4, 8, 9, 14].

## 📊 Conjuntos de Dados Utilizados
[cite_start]Os experimentos foram rodados sob as mesmas partições (60% Treino / 20% Validação / 20% Teste) nas seguintes bases públicas da UCI[cite: 25, 26, 34]:
1. **Iris:** Classificação Multiclasse (Perfeitamente Balanceado).
2. **Breast Cancer Wisconsin:** Classificação Binária (Levemente Desbalanceado).
3. **Wine Quality (Red):** Regressão Quantitativa (Distribuição Unimodal).
4. **Heart Disease (Statlog):** Classificação Binária (Atributos Numéricos e Categóricos Mistos).

---

## 🛠️ Requisitos e Configuração do Ambiente

O projeto foi isolado em um ambiente virtual do Python para garantir que as versões internas das dependências de grafos e versionamento do `scikit-fuzzy` funcionassem sem quebra de compatibilidade.

### 1. Clonando ou acessando a pasta do projeto:
```bash
cd "C:/Users/geova/Documents/mestrado/trabaçho final"


 Ativando o Ambiente Virtual (.venv):No Windows (PowerShell):PowerShell& ".venv/Scripts/Activate.ps1"
No Windows (Prompt de Comando):DOS.venv\Scripts\activate.bat
3. Instalação das Dependências:Caso precise reconstruir o ambiente do zero, instale os pacotes principais e as dependências internas do compilador Fuzzy:Bashpip install numpy pandas scikit-learn scikit-fuzzy packaging networkx
🚀 Como Executar os ExperimentosO código fonte está organizado na pasta src/. Os scripts executam automaticamente o loop de 21 rodadas independentes utilizando sementes amostrais alteradas incrementalmente para anular o viés de estocasticidade (inicialização aleatória de pesos).  Parte 1: Executar Redes Neurais (Iris e Breast Cancer)Roda os modelos de classificação das RNAs (Configuração Rasa vs. Profunda):Bashpython "src/Loop dos Experimentos.py"
Parte 2: Executar Sistemas Fuzzy (Iris e Breast Cancer)Aplica a seleção de características via ANOVA e executa o motor de inferência Mamdani (Triangular vs. Gaussiano):Bashpython "src/Experimentos Fuzzy.py"
Parte 3: Executar Scripts Unificados Finais (Wine Quality e Heart Disease)Faz o download em tempo real das duas últimas bases e computa as métricas de classificação e regressão (MSE, MAE, $R^2$ e Acurácia):  Bashpython "src/experimentos_finais.py"
├── .vscode/                     # Configurações locais do editor VS Code
├── tstex_modules/               # Módulos e pacotes auxiliares de compilação LaTeX
├── .gitignore                   # Arquivo de exclusão de arquivos temporários do Git
├── readme.md                    # Este guia explicativo de reprodução e execução
├── IEEE_Journal_Paper_Template/ # Pasta contendo o artigo científico estruturado
│   └── relatorio_ic_geovane.tex # Seu arquivo-fonte principal do relatório (IEEE)
└── src/                         # Código-fonte executável do projeto
    ├── Loop dos Experimentos.py # Pipeline e loops das RNAs iniciais
    ├── Experimentos Fuzzy.py    # Pipeline dos modelos Fuzzy iniciais
    └── experimentos_finais.py   # Download e inferência dos datasets finais