Aqui está a **Análise Exploratória Básica** estruturada para cada um dos 4 conjuntos de dados selecionados da UCI. Você pode copiar e colar este bloco diretamente na seção correspondente do seu artigo no formato IEEE , pois ele já atende a todos os critérios exigidos pelo professor (descrição do problema, número de amostras, número de atributos, tipo de tarefa, variável de saída e análise de balanceamento).

---

## 📊 Seção: Análise Exploratória dos Dados

1. Iris Flower Dataset 

* **Descrição do Problema:** Trata-se de um problema clássico de botânica onde o objetivo é identificar a espécie de uma planta do gênero Íris com base nas dimensões físicas de suas pétalas e sépalas.
* 
**Tipo de Tarefa:** Classificação Multiclasse.


* 
**Número de Amostras ($N$):** 150 amostras.


* 
**Número de Atributos:** 4 atributos preditivos numéricos (contínuos, em cm):


1. Comprimento da sépala (`sepal length`)
2. Largura da sépala (`sepal width`)
3. Comprimento da pétala (`petal length`)
4. Largura da pétala (`petal width`)


* 
**Variável de Saída (Target):** Espécie da planta (Classe).


* **Análise de Balanceamento:** **Perfeitamente balanceado**. O dataset possui exatamente 50 amostras para cada uma das 3 classes (`Iris-setosa`, `Iris-versicolor` e `Iris-virginica`), totalizando uma distribuição de 33,3% para cada. Não há necessidade de técnicas de reamostragem.

---

2. Breast Cancer Wisconsin (Diagnostic) 

* **Descrição do Problema:** Análise computacional para o diagnóstico clínico de câncer de mama. O objetivo é prever se uma massa tumoral aspirada por agulha fina é maligna ou benigna com base em características geométricas e texturais do núcleo celular.
* 
**Tipo de Tarefa:** Classificação Binária.


* 
**Número de Amostras ($N$):** 569 amostras.


* 
**Número de Atributos:** 30 atributos preditivos numéricos (contínuos, representando valores médios, erros-padrão e os piores valores de características como raio, textura, perímetro, área, suavidade, concavidade, etc.).


* 
**Variável de Saída (Target):** Diagnóstico clínico  (`Maligno` ou `Benigno`).


* **Análise de Balanceamento:** **Levemente desbalanceado**. O conjunto contém 357 amostras benignas (62,7%) e 212 amostras malignas (37,3%). Por ser um desbalanceamento sutil, os modelos de RNAs e Neuro-Fuzzy conseguem aprender sem a necessidade estrita de algoritmos de balanceamento (como SMOTE), mas exige atenção redobrada à métrica de **F1-Score** e **Revocação** (sensibilidade) na matriz de confusão.



---

3. Wine Quality Dataset (Vinho Tinto) 

* **Descrição do Problema:** Avaliação da qualidade sensorial de vinhos de mesa produzidos no norte de Portugal (Vinho Verde). O objetivo é modelar a percepção humana de qualidade a partir de medições físico-químicas laboratoriais do produto.
* 
**Tipo de Tarefa:** Regressão ou Classificação Ordinal. (Recomenda-se tratar como Regressão para testar as métricas MSE, RMSE e MAE exigidas no trabalho ).


* 
**Número de Amostras ($N$):** 1.599 amostras.


* 
**Número de Atributos:** 11 atributos preditivos numéricos (contínuos e discretos, como acidez fixa, acidez volátil, ácido cítrico, açúcar residual, cloretos, dióxido de enxofre livre, dióxido de enxofre total, densidade, pH, sulfatos e teor alcoólico).


* 
**Variável de Saída (Target):** Qualidade do vinho (nota contínua/inteira de 0 a 10).


* **Análise de Distribuição:** **Distribuição Normal/Sino**. A maioria esmagadora dos vinhos concentra-se nas notas intermediárias 5 e 6 (mais de 80% dos dados). Existem pouquíssimos registros de vinhos excelentes (notas 7 ou 8) ou muito ruins (notas 3 ou 4). Esse comportamento desafiará as RNAs e os sistemas Neuro-Fuzzy a não convergirem apenas para a média do mercado.

---

4. Heart Disease Dataset (Statlog) 

* **Descrição do Problema:** Diagnóstico de presença ou ausência de doença arterial coronariana em pacientes cardíacos, combinando dados demográficos, sintomas clínicos e exames laboratoriais (como eletrocardiograma e teste de esforço).
* 
**Tipo de Tarefa:** Classificação Binária.


* 
**Número de Amostras ($N$):** 270 amostras.


* 
**Número de Atributos:** 13 atributos preditivos mistos (inclui variáveis contínuas como idade, pressão arterial em repouso, colesterol, frequência cardíaca máxima e variáveis categóricas/nominais como sexo, tipo de dor no peito, açúcar no sangue em jejum, resultados do ECG, etc.).


* 
**Variável de Saída (Target):** Presença ou ausência de doença cardíaca.


* **Análise de Balanceamento:** **Bem balanceado**. Contém 150 amostras de pacientes saudáveis (55,6%) e 120 amostras de pacientes com a doença (44,4%). É uma base excelente e limpa para testar a estabilidade dos modelos Fuzzy em classificar dados de naturezas distintas (numéricos e categóricos misturados).

---

### 💡 Próximo Passo Prático para o Grupo:
