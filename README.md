#  Análise de Desempenho Acadêmico em Curso EaD com Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

> Será que dá para prever se um aluno vai ser aprovado **sem olhar para as notas**?  
> Esse projeto tenta responder essa pergunta usando apenas os dados de acesso à plataforma EaD.

---

##  Sumário

- [Contexto](#-contexto)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Visualização de Dados](#-visualização-de-dados)
- [Modelos de Classificação](#-modelos-de-classificação)
- [Modelos de Regressão](#-modelos-de-regressão)
- [Análise de Agrupamento](#-análise-de-agrupamento)
- [Principais Descobertas](#-principais-descobertas)
- [Tecnologias](#-tecnologias)
- [Dados](#-dados)

---

##  Contexto

Durante a pandemia de COVID-19, o **IFNMG** ofertou o curso de
**Programador de Dispositivos Móveis (PDM)** em 2 edições a distância,
com mais de **1.000 alunos** matriculados.

No ambiente EaD, professores enfrentam um desafio crítico: identificar
alunos em risco de reprovação sem o contato presencial. Este projeto
aplica técnicas de **Visualização de Dados** e **Inteligência Artificial**
para encontrar padrões no comportamento de acesso à plataforma que
possam indicar o desempenho acadêmico do aluno.

---

##  Estrutura do Projeto
├── Visualizacao de dados.py   # análise exploratória e gráficos

├── Treinamento IA.py          # modelos de machine learning

├── assets/                    # imagens geradas

│   ├── barras_resultado.png

│   ├── boxplot_notas.png

│   ├── matrizes_confusao.png

│   ├── regressao_real_vs_predito.png

│   ├── importancia_variaveis.png

│   ├── importancia_gb.png

│   ├── clustering_pca.png

│   └── elbow_chart.png

└── README.md

---

##  Visualização de Dados

### 1. Importação e Pré-processamento

```python
df = pd.read_csv(r"caminho\dados.csv")
df = df[df['Resultado'].isin(['APROVADO', 'REPROVADO'])].copy()
```
Carrega o arquivo CSV e remove linhas com resultados inválidos ou vazios,
mantendo apenas alunos com resultado definido.

```python
for c in nota_cols:
    df[c] = pd.to_numeric(df[c].replace('-', np.nan), errors='coerce')
```
As colunas de nota contêm o caractere `-` onde o aluno não fez a prova.
Esse trecho substitui `-` por `NaN` (valor nulo) e converte tudo para número.

```python
df['Resultado_num'] = (df['Resultado'] == 'APROVADO').astype(int)
```
Cria uma coluna numérica onde `APROVADO = 1` e `REPROVADO = 0`,
necessária para os algoritmos de Machine Learning.

```python
scaler = MinMaxScaler()
df_norm[num_cols] = scaler.fit_transform(df[num_cols].fillna(0))
```
Normaliza todas as colunas numéricas para uma escala entre 0 e 1,
evitando que variáveis com valores muito altos dominem os modelos.

---

### 2. Gráficos de Barra — Aprovados e Reprovados

```python
contagem = df['Resultado'].value_counts()
barras = ax.bar(contagem.index, contagem.values, color=[...])
```
Conta quantos alunos foram aprovados e reprovados e gera um gráfico
de barras para cada disciplina, com cores diferentes para cada resultado
e o valor absoluto exibido acima de cada barra.

![Barras](assets/barras_resultado.png)

---

### 3. Box Plots das Notas

```python
aprov = df[df['Resultado']=='APROVADO'][col].dropna()
reprov = df[df['Resultado']=='REPROVADO'][col].dropna()
bp = ax.boxplot([aprov, reprov], tick_labels=['Aprovados','Reprovados'], patch_artist=True)
```
Separa as notas de aprovados e reprovados e gera um box plot para cada
disciplina. O box plot mostra a mediana, os quartis e os valores extremos,
permitindo comparar a distribuição das notas entre os dois grupos.
A disciplina C apresentou maior variação entre os aprovados.

![Boxplot](assets/boxplot_notas.png)

---

### 4. Engenharia de Feature — Total de Acessos ao Fórum

```python
df['Total_Forum'] = (df['Fórum Geral'] + df['Fórum A'] +
                     df['Fórum B'] + df['Fórum C'])
```
Cria uma nova variável somando todos os acessos aos fóruns do curso.
Essa técnica se chama **engenharia de features** — criar novas variáveis
a partir das existentes para capturar informações mais relevantes.

```python
ax.scatter(reprov['Total_Forum'], reprov['NotaTotal'], ...)
ax.scatter(aprov['Total_Forum'], aprov['NotaTotal'], ...)
```
Gera um gráfico de dispersão relacionando o engajamento nos fóruns
com a nota total, separando aprovados e reprovados por cor e marcador.

```python
medias = df.groupby('Resultado')['Total_Forum'].mean()
```
Calcula a média de acessos ao fórum por grupo para comparação direta
entre aprovados e reprovados.

---

##  Modelos de Classificação

### 1. Preparação dos Dados

```python
acesso_cols = ['Fórum Geral', 'Página Geral', 'Arquivo Conteudo', ...]
X = df[acesso_cols]
y_clf = df['Resultado_num']
```
Define as variáveis descritivas (`X`) como apenas os dados de acesso
à plataforma, sem incluir nenhuma nota. A variável alvo (`y`) é o
resultado final do aluno.

```python
X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.30, random_state=42, stratify=y_clf
)
```
Divide os dados em 70% para treino e 30% para teste. O parâmetro
`stratify` garante que a proporção de aprovados e reprovados seja
mantida igual nos dois conjuntos. O `random_state=42` garante que
a divisão seja sempre a mesma, tornando os resultados reproduzíveis.

---

### 2. Árvore de Decisão

```python
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_clf_train)
y_pred_dt = dt.predict(X_test)
```
Treina uma Árvore de Decisão com profundidade máxima de 5 níveis.
O modelo aprende regras do tipo *"se o aluno acessou a Tarefa A mais
de X vezes, então..."* para classificar cada aluno. O `max_depth=5`
evita que a árvore fique complexa demais e decore os dados de treino.

**Resultado: 84% de acurácia**

---

### 3. Random Forest

```python
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
rf.fit(X_train, y_clf_train)
y_pred_rf = rf.predict(X_test)
```
Treina um conjunto de 100 árvores de decisão diferentes (`n_estimators=100`),
cada uma treinada com uma amostra aleatória dos dados. A classificação
final é feita por votação majoritária entre todas as árvores, o que
torna o modelo mais robusto e preciso que uma única árvore.

**Resultado: 89% de acurácia** ✅

### Matrizes de Confusão

```python
cm = confusion_matrix(y_clf_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
```
Gera uma matriz que mostra quantos alunos foram classificados
corretamente e quais foram confundidos pelo modelo — aprovados
classificados como reprovados e vice-versa.

![Matrizes de Confusão](assets/matrizes_confusao.png)

### Importância das Variáveis

```python
importancias = pd.Series(rf.feature_importances_, index=acesso_cols).sort_values()
```
O Random Forest calcula automaticamente quais variáveis foram mais
relevantes para as decisões. Isso permite identificar quais tipos
de acesso à plataforma mais influenciam no resultado do aluno.

![Importância RF](assets/importancia_variaveis.png)

---

##  Modelos de Regressão

### 1. Regressão Linear

```python
lr = LinearRegression()
lr.fit(X_train, y_reg_train)
y_pred_lr = lr.predict(X_test)
```
Tenta encontrar uma relação linear entre os acessos à plataforma
e a nota final. O modelo aprende um peso para cada variável de acesso
e soma tudo para estimar a nota.

### 2. Gradient Boosting

```python
gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
gb.fit(X_train, y_reg_train)
y_pred_gb = gb.predict(X_test)
```
Constrói 200 árvores de forma sequencial, onde cada nova árvore
tenta corrigir os erros da anterior. O `learning_rate=0.05` faz
esse processo ser gradual e cuidadoso para evitar overfitting.

### Avaliação

```python
def avaliar_regressao(nome, y_real, y_pred):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)
```
- **R²:** indica o quanto o modelo explica a variação das notas (1.0 = perfeito, 0 = ruim)
- **RMSE:** erro médio em pontos de nota — quanto menor, melhor
- **MAE:** erro absoluto médio

| Modelo | R² | RMSE |
|--------|----|------|
| Regressão Linear | -0.05 | 97.23 |
| Gradient Boosting | -0.12 | 100.54 |

![Regressão](assets/regressao_real_vs_predito.png)

![Importância GB](assets/importancia_gb.png)

> ⚠️ **Limitação:** os modelos de regressão apresentaram R² negativo,
> indicando que os dados de acesso isoladamente não são suficientes para
> estimar a nota final com precisão. Isso reforça que o desempenho acadêmico
> depende de fatores além do engajamento na plataforma — o que é, por si só,
> uma descoberta relevante.

---

##  Análise de Agrupamento

### 1. Padronização

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
Padroniza as variáveis para que todas tenham média 0 e desvio padrão 1.
Diferente da normalização, a padronização é mais adequada para PCA
pois trata variáveis com escalas muito diferentes de forma justa.

### 2. Redução de Dimensionalidade com PCA

```python
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
```
Reduz as 25 variáveis de acesso para apenas 2 componentes principais,
mantendo o máximo de informação possível. Isso permite visualizar
os alunos em um gráfico 2D e facilita o agrupamento.

- PC1 explica **30.7%** da variância
- PC2 explica **13.8%** da variância
- Total: **44.5%** da informação original preservada

### 3. Agrupamento com K-Means

```python
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
clusters = kmeans.fit_predict(X_pca)
```
Agrupa os alunos em 3 grupos baseado na similaridade do comportamento
de acesso. O algoritmo encontra os 3 centros que minimizam a distância
de cada aluno ao seu grupo. O `n_init=20` roda o algoritmo 20 vezes
com pontos de partida diferentes para garantir o melhor resultado.

### 4. Método do Cotovelo

```python
for k in ks:
    inercias.append(KMeans(n_clusters=k, n_init=20).fit(X_pca).inertia_)
```
Testa diferentes valores de k (de 2 a 8) e mede a inércia de cada um.
O gráfico resultante forma um "cotovelo" que indica o k ideal —
neste caso, **k=3**.

| Cluster | Alunos | % Aprovados | Perfil |
|---------|--------|-------------|--------|
| 0 | 479 | 12.9% |  Baixo engajamento    |
| 1 | 306 | 84.6% |    Alto desempenho    |
| 2 | 232 | 77.2% |      Intermediário    |

![Clustering](assets/clustering_pca.png)

![Elbow](assets/elbow_chart.png)

---

##  Principais Descobertas

-  É possível prever a aprovação de um aluno com **89% de acurácia**
usando apenas dados de acesso à plataforma, sem considerar as notas
-  Alunos com maior engajamento nos fóruns tendem a ter notas mais altas
-  Os alunos se agrupam naturalmente em 3 perfis distintos de engajamento
-  O volume de acessos isolado não é suficiente para estimar a nota exata —
o desempenho acadêmico é multifatorial

---

## 🛠 Tecnologias

| Ferramenta | Uso |
|------------|-----|
| Python 3.11 | Linguagem principal |
| Pandas | Manipulação de dados |
| Scikit-learn | Modelos de ML |
| Matplotlib | Visualizações |

---

## Dados

Os dados utilizados são de alunos reais anonimizados, fornecidos pelo **IFNMG**.
O arquivo CSV não está disponível neste repositório por questões de privacidade
e proteção de dados (LGPD).

---

*Trabalho prático das disciplinas de Visualização de Dados e Inteligência Artificial — Curso Técnico IFNMG*