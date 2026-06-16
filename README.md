# 🎓 Análise de Desempenho Acadêmico em Curso EaD com Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

> Será que dá para prever se um aluno vai ser aprovado **sem olhar para as notas**?  
> Esse projeto tenta responder essa pergunta usando apenas os dados de acesso à plataforma EaD.

---

## 📋 Sumário

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

## 📌 Contexto

Durante a pandemia de COVID-19, o **IFNMG** ofertou o curso de
**Programador de Dispositivos Móveis (PDM)** em 2 edições a distância,
com mais de **1.000 alunos** matriculados.

No ambiente EaD, professores enfrentam um desafio crítico: identificar
alunos em risco de reprovação sem o contato presencial. Este projeto
aplica técnicas de **Visualização de Dados** e **Inteligência Artificial**
para encontrar padrões no comportamento de acesso à plataforma que
possam indicar o desempenho acadêmico do aluno.

---

## 📁 Estrutura do Projeto
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

## 📊 Visualização de Dados

### Aprovados e Reprovados por Disciplina

Comparação do resultado final dos alunos nas três disciplinas do curso.

![Barras](assets/barras_resultado.png)

---

### Distribuição de Notas por Disciplina

Box plots comparando a distribuição das notas nas disciplinas A, B e C.
A disciplina C apresentou maior variação entre os aprovados.

![Boxplot](assets/boxplot_notas.png)

---

### Engajamento nos Fóruns vs Desempenho

Uma das descobertas mais interessantes do projeto: alunos que participam
mais dos fóruns tendem a ter notas significativamente maiores.

---

## 🤖 Modelos de Classificação

**Variável alvo:** `Resultado` (APROVADO / REPROVADO)  
**Variáveis descritivas:** apenas dados de acesso à plataforma — sem notas  
**Divisão:** 70% treino / 30% teste

| Modelo | Acurácia | Observação |
|--------|----------|------------|
| Árvore de Decisão | 84% | Modelo interpretável, baseado em regras |
| **Random Forest** | **89%** | ✅ Melhor modelo — ensemble de árvores |

### Matrizes de Confusão

![Matrizes de Confusão](assets/matrizes_confusao.png)

### Importância das Variáveis — Random Forest

Quais acessos mais influenciam na previsão de aprovação?

![Importância RF](assets/importancia_variaveis.png)

---

## 📈 Modelos de Regressão

**Variável alvo:** `NotaTotal`  
**Variáveis descritivas:** apenas dados de acesso à plataforma — sem notas  
**Divisão:** 70% treino / 30% teste

| Modelo | R² | RMSE |
|--------|----|------|
| Regressão Linear | -0.05 | 97.23 |
| Gradient Boosting | -0.12 | 100.54 |

### Real vs Predito

![Regressão](assets/regressao_real_vs_predito.png)

### Importância das Variáveis — Gradient Boosting

![Importância GB](assets/importancia_gb.png)

> ⚠️ **Limitação:** os modelos de regressão apresentaram R² negativo,
> indicando que os dados de acesso isoladamente não são suficientes para
> estimar a nota final com precisão. Isso reforça que o desempenho acadêmico
> depende de fatores além do engajamento na plataforma — o que é, por si só,
> uma descoberta relevante.

---

## 🔵 Análise de Agrupamento

Redução de dimensionalidade com **PCA** seguida de agrupamento com **K-Means**
para identificar perfis distintos de alunos.

- PC1 explica **30.7%** da variância
- PC2 explica **13.8%** da variância
- Total explicado: **44.5%**

| Cluster | Alunos | % Aprovados | Perfil |
|---------|--------|-------------|--------|
| 0 | 479 | 12.9% | 🔴 Baixo engajamento e desempenho |
| 1 | 306 | 84.6% | 🟢 Alto desempenho |
| 2 | 232 | 77.2% | 🟡 Engajamento intermediário |

### Clusters — PCA + K-Means

![Clustering](assets/clustering_pca.png)

### Método do Cotovelo — Justificativa para k=3

![Elbow](assets/elbow_chart.png)

---

## 💡 Principais Descobertas

- ✅ É possível prever a aprovação de um aluno com **89% de acurácia**
usando apenas dados de acesso à plataforma, sem considerar as notas
- 📊 Alunos com maior engajamento nos fóruns tendem a ter notas mais altas
- 🔵 Os alunos se agrupam naturalmente em 3 perfis distintos de engajamento
- ⚠️ O volume de acessos isolado não é suficiente para estimar a nota exata —
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

## 🔒 Dados

Os dados utilizados são de alunos reais anonimizados, fornecidos pelo **IFNMG**.
O arquivo CSV não está disponível neste repositório por questões de privacidade
e proteção de dados (LGPD).

---

*Trabalho prático das disciplinas de Visualização de Dados e Inteligência Artificial — Curso Técnico IFNMG*