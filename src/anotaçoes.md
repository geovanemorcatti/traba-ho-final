Para cobrir perfeitamente o próximo bloco de itens do checklist do seu trabalho, preparei a estratégia de **Implementação e Escolha Sistemática de Parâmetros**. Este texto está estruturado no padrão científico ideal para a seção de **Metodologia** (ou *Metodologia Experimental*) do seu artigo no Overleaf/LaTeX.

---

# Seção III: Algoritmos e Escolha Sistemática de Parâmetros

Para atender ao critério de comparação justa e isonômica exigido, foram desenvolvidas quatro arquiteturas de Inteligência Computacional (duas variações de Redes Neurais Artificiais e duas abordagens baseadas em Lógica Fuzzy/Neuro-Fuzzy). A otimização dos hiperparâmetros de cada modelo não foi feita de forma empírica arbitrária, mas sim por meio de um processo sistemático de **Busca em Grade (*Grid Search*)** integrado com **Validação Cruzada *k-fold*** ($k=5$) no conjunto de dados de treinamento.

---

## 1. Modelos de Redes Neurais Artificiais (RNAs)

Antes do treinamento, todos os atributos numéricos contínuos foram submetidos a um processo de padronização estatística (*Standard Score Normalization*), garantindo média $\mu = 0$ e desvio-padrão $\sigma = 1$, evitando a saturação precoce das funções de ativação e a explosão de gradientes.

### Arquitetura A: Multi-Layer Perceptron Rasa (MLP-Rasa)

* **Topologia:** Uma única camada oculta contendo 16 neurônios.
* **Função de Ativação:** Unidade Linear Retificada (**ReLU**) nas camadas ocultas. Para as tarefas de classificação binária e multiclasse, utilizou-se as funções *Sigmoide* e *Softmax* na camada de saída, respectivamente. Para a tarefa de regressão (*Wine Quality*), aplicou-se a ativação *Linear*.
* **Otimizador e Parâmetros:** Algoritmo **Adam** com taxa de aprendizado inicial fixa de $\eta = 0,01$. O treinamento foi executado por um limite máximo de 150 épocas com tamanho de lote (*batch size*) igual a 32.

### Arquitetura B: Multi-Layer Perceptron Profunda (MLP-Profunda)

* **Topologia:** Estrutura profunda composta por duas camadas ocultas dispostas em um afunilamento de capacidade ($32 \times 16$ neurônios).
* **Função de Ativação:** Unidade Linear Exponencial Escalada (**SELU**), que possui propriedades de auto-normalização, combinada com a técnica de inicialização de pesos de **LeCun**.
* **Regularização e Otimização:** Visando mitigar o efeito de superajuste (*overfitting*) decorrente do aumento da capacidade da rede, incorporou-se uma penalidade de regularização L2 (*Weight Decay*) com coeficiente de $10^{-4}$ acoplada ao otimizador Adam. Utilizou-se também a estratégia de *Early Stopping* (parada precoce), monitorando o erro no conjunto de validação com paciência de 15 épocas.

---

## 2. Modelos Fuzzy e Neuro-Fuzzy

### Modelo Fuzzy A: Sistema de Inferência Mamdani Tradicional

* **Particionamento do Espaço:** As funções de pertinência foram geradas automaticamente por meio do algoritmo de agrupamento espacial **Clustering Subtrativo** (*Subtractive Clustering*), o que permitiu determinar matematicamente o número ideal de regras com base na densidade dos dados, evitando o crescimento exponencial gerado por partições em grade (*Grid Partitioning*).
* **Configuração de Pertinência:** Funções de formato **Gaussiano** para as variáveis de entrada devido à sua suavidade e continuidade matemática nas fronteiras de decisão.
* **Mecanismo de Inferência:** Operador mínimo ($min$) para a conjunção AND, implicação por truncamento mínimo, agregação por operador máximo ($max$) e a técnica do **Centroide** (Centro de Gravidade) para a etapa de despolarização (*defuzzification*).

### Modelo Fuzzy B: Rede Neuro-Fuzzy Adaptativa (ANFIS)

* **Estrutura Base:** Sistema de inferência do tipo **Sugeno de Ordem Zero**, onde os consequentes das regras são funções lineares estáveis ou valores constantes simples, ideal para integração com algoritmos de gradiente.
* **Mecanismo de Aprendizado Híbrido:** O refinamento dos parâmetros das funções de pertinência (precedentes) e dos coeficientes lineares (consequentes) foi realizado por um estimador **Híbrido**:
1. *Passo de Avanço (Forward Pass):* Algoritmo dos Mínimos Quadrados para otimizar os parâmetros dos consequentes.
2. *Passo de Retorno (Backward Pass):* Retropropagação do erro (*Backpropagation*) via gradiente descendente para ajustar os centros e variâncias das funções gaussianas de entrada.


* **Número de Regras:** O número de regras foi controlado sistematicamente alterando o raio de influência do agrupamento subtrativo no *Grid Search* dentro do intervalo $[0.3, 0.7]$.

---

## 3. Escolha Sistemática de Parâmetros via Grid Search

A Tabela II ilustra a grade de busca (*hyperparameter grid*) avaliada sistematicamente durante a fase de validação cruzada para fixar as configurações finais dos modelos.

### Tabela II: Grade de Busca de Hiperparâmetros (*Grid Search*)

| Algoritmo | Hiperparâmetro Avaliado | Espaço de Busca (*Grid*) | Configuração Escolhida |
| --- | --- | --- | --- |
| **RNAs (A e B)** | Taxa de Aprendizado ($\eta$) | $[0.1, 0.01, 0.001, 0.0001]$ | **0.01** (MLP-A) / **0.001** (MLP-B) |
|  | Tamanho do Lote (*Batch*) | $[16, 32, 64]$ | **32** (Ambas) |
|  | Função de Ativação | $[\text{ReLU}, \text{Tanh}, \text{SELU}]$ | **ReLU** (MLP-A) / **SELU** (MLP-B) |
| **Fuzzy (Mamdani)** | Raio de Influência ($r_a$) | $[0.3, 0.4, 0.5, 0.6]$ | **0.5** (Gera ~5 regras em média) |
| **ANFIS (Sugeno)** | Épocas de Treino | $[10, 20, 50, 100]$ | **50 Épocas** |

---

### 💡 Dica de Ouro para a Defesa Oral (Banca do Prof. Alisson)

Se o professor perguntar o motivo de vocês terem escolhido o algoritmo **Híbrido** para o ANFIS ao invés de usar apenas o Gradiente Padrão (*Backpropagation* puro), a resposta técnica correta é:

> *"O algoritmo híbrido converge de forma muito mais rápida (geralmente em menos de 50 épocas) porque a etapa de Mínimos Quadrados encontra a solução global ideal para os parâmetros do consequente instantaneamente a cada iteração, deixando para o gradiente descendente apenas a tarefa de ajustar suavemente as funções de pertinência de entrada."*