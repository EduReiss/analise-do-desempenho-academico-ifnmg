import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Carrega os dados
df = pd.read_csv(r"C:\Users\mateu\Desktop\Programação\Projetos\Projeto multidisciplinar IFNMG\dados.csv")

# Limpeza básica
df = df[df['Resultado'].isin(['APROVADO', 'REPROVADO'])].copy()

# Substituir '-' por NaN e converter notas para numérico
nota_cols = ['A_NotaRegular','A_NotaRecuperacao','B_NotaRegular',
             'B_NotaRecuperacao','C_NotaRegular','C_NotaRecuperacao','NotaTotal']
for c in nota_cols:
    df[c] = pd.to_numeric(df[c].astype(str).str.strip().replace('-', np.nan), errors='coerce')

# Colunas de acesso (variáveis descritivas — sem notas)
acesso_cols = [
    'Fórum Geral', 'Página Geral', 'Arquivo Conteudo', 'Relatório  ',
    'URL Geral', 'Arquivo A', 'Tarefa A', 'URL A', 'Fórum A',
    'Página A', 'Pasta A', 'Questionário A',
    'Arquivo B', 'Tarefa B', 'URL B', 'Fórum B', 'Pasta B',
    'Questionário B', 'Chat B',
    'Arquivo C', 'Tarefa C', 'URL C', 'Fórum C',
    'Questionário C', 'Chat C'
]

# Garantir que colunas de acesso são numéricas e sem NaN
df[acesso_cols] = df[acesso_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# Variável alvo para classificação
le = LabelEncoder()
df['Resultado_num'] = le.fit_transform(df['Resultado'])  # APROVADO=0, REPROVADO=1

X = df[acesso_cols]
y_clf = df['Resultado_num']          # alvo classificação
y_reg = df['NotaTotal'].fillna(0)    # alvo regressão

# Divisão 70% treino / 30% teste
X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.30, random_state=42, stratify=y_clf
)

_, _, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.30, random_state=42
)

print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste:  {X_test.shape[0]} amostras")
print(f"Classes no treino: {y_clf_train.value_counts().to_dict()}")

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

# MODELO 1: Árvore de Decisão
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_clf_train)
y_pred_dt = dt.predict(X_test)

print("=" * 50)
print("MODELO 1 — Árvore de Decisão")
print("=" * 50)
print(f"Acurácia: {accuracy_score(y_clf_test, y_pred_dt):.4f}")
print(classification_report(y_clf_test, y_pred_dt,
                             target_names=le.classes_))

# MODELO 2: Random Forest 
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
rf.fit(X_train, y_clf_train)
y_pred_rf = rf.predict(X_test)

print("=" * 50)
print("MODELO 2 — Random Forest")
print("=" * 50)
print(f"Acurácia: {accuracy_score(y_clf_test, y_pred_rf):.4f}")
print(classification_report(y_clf_test, y_pred_rf,
                             target_names=le.classes_))

#Matrizes de confusão lado a lado 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, y_pred, titulo in zip(
    axes,
    [y_pred_dt, y_pred_rf],
    ['Árvore de Decisão', 'Random Forest']
):
    cm = confusion_matrix(y_clf_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(titulo, fontsize=13)

plt.tight_layout()
plt.savefig('matrizes_confusao.png', dpi=150)
plt.show()

# Importância das variáveis (Random Forest) 
importancias = pd.Series(rf.feature_importances_, index=acesso_cols).sort_values()

plt.figure(figsize=(8, 7))
importancias.plot(kind='barh', color='#185FA5')
plt.title('Importância das variáveis — Random Forest', fontsize=13)
plt.xlabel('Importância')
plt.tight_layout()
plt.savefig('importancia_variaveis.png', dpi=150)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def avaliar_regressao(nome, y_real, y_pred):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae  = mean_absolute_error(y_real, y_pred)
    r2   = r2_score(y_real, y_pred)
    print(f"\n{'='*50}")
    print(f"{nome}")
    print(f"{'='*50}")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    return y_pred

# MODELO 1: Regressão Linear 
lr = LinearRegression()
lr.fit(X_train, y_reg_train)
y_pred_lr = avaliar_regressao(
    "MODELO 1 — Regressão Linear",
    y_reg_test, lr.predict(X_test)
)

#  MODELO 2: Gradient Boosting Regressor 
gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                learning_rate=0.05, random_state=42)
gb.fit(X_train, y_reg_train)
y_pred_gb = avaliar_regressao(
    "MODELO 2 — Gradient Boosting",
    y_reg_test, gb.predict(X_test)
)

#  Gráficos: real vs predito 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, y_pred, titulo, cor in zip(
    axes,
    [y_pred_lr, y_pred_gb],
    ['Regressão Linear', 'Gradient Boosting'],
    ['#185FA5', '#1D9E75']
):
    ax.scatter(y_reg_test, y_pred, alpha=0.4, s=20, color=cor)
    lim = [min(y_reg_test.min(), y_pred.min()) - 5,
           max(y_reg_test.max(), y_pred.max()) + 5]
    ax.plot(lim, lim, 'r--', lw=1.5, label='Perfeito')
    ax.set_xlabel('Nota Total real')
    ax.set_ylabel('Nota Total predita')
    ax.set_title(titulo, fontsize=13)
    ax.legend()

plt.tight_layout()
plt.savefig('regressao_real_vs_predito.png', dpi=150)
plt.show()

# Importância das variáveis (Gradient Boosting) 
imp_gb = pd.Series(gb.feature_importances_, index=acesso_cols).sort_values()

plt.figure(figsize=(8, 7))
imp_gb.plot(kind='barh', color='#1D9E75')
plt.title('Importância das variáveis — Gradient Boosting', fontsize=13)
plt.xlabel('Importância')
plt.tight_layout()
plt.savefig('importancia_gb.png', dpi=150)
plt.show()

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Pré-processamento: padronizar variáveis de acesso 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Redução de dimensionalidade com PCA (2 componentes) 
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

var_exp = pca.explained_variance_ratio_ * 100
print(f"Variância explicada — PC1: {var_exp[0]:.1f}%  |  PC2: {var_exp[1]:.1f}%")
print(f"Total explicado: {sum(var_exp):.1f}%")

# Agrupamento com K-Means (3 grupos) 
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
clusters = kmeans.fit_predict(X_pca)

df['Cluster'] = clusters

#  Perfil de cada cluster 
perfil = df.groupby('Cluster').agg(
    n=('Resultado', 'count'),
    pct_aprovado=('Resultado', lambda x: (x == 'APROVADO').mean() * 100),
    nota_media=('NotaTotal', 'mean'),
    acesso_medio=(acesso_cols[0], 'mean')   # Fórum Geral como exemplo
).round(1)
print("\nPerfil dos clusters:")
print(perfil)

# Visualizações 
cores_cluster = ['#185FA5', '#1D9E75', '#D85A30']
marcadores_res = {'APROVADO': 'o', 'REPROVADO': '^'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Painel 1:  clusters coloridos por grupo
ax1 = axes[0]
for k, cor in enumerate(cores_cluster):
    mask = clusters == k
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=cor, alpha=0.55, s=25, label=f'Cluster {k}')
# centróides
cx = kmeans.cluster_centers_
ax1.scatter(cx[:, 0], cx[:, 1], c='black', s=120, marker='X',
            zorder=5, label='Centróides')
ax1.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var.)')
ax1.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var.)')
ax1.set_title('Agrupamento K-Means (3 clusters) — PCA', fontsize=13)
ax1.legend(fontsize=9)

# Painel 2 : mesmos pontos coloridos por Resultado
ax2 = axes[1]
cores_res = {'APROVADO': '#1D9E75', 'REPROVADO': '#D85A30'}
for res, grupo in df.groupby('Resultado'):
    idx = grupo.index
    mask = df.index.isin(idx)
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=cores_res[res], alpha=0.45, s=20,
                marker=marcadores_res[res], label=res)
ax2.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var.)')
ax2.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var.)')
ax2.set_title('Mesmo espaço PCA — colorido por Resultado', fontsize=13)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('clustering_pca.png', dpi=150)
plt.show()

# Elbow chart (para justificar k=3)
inercias = []
ks = range(2, 9)
for k in ks:
    inercias.append(KMeans(n_clusters=k, n_init=20,
                           random_state=42).fit(X_pca).inertia_)

plt.figure(figsize=(7, 4))
plt.plot(ks, inercias, 'o-', color='#185FA5', lw=2)
plt.axvline(3, color='#D85A30', linestyle='--', label='k=3 escolhido')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do cotovelo — K-Means', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig('elbow_chart.png', dpi=150)
plt.show()

print("\n=== RESULTADOS DOS MODELOS ===")

# Classificação
print(f"\nÁrvore de Decisão - Acurácia: {accuracy_score(y_clf_test, y_pred_dt):.4f}")
print(f"Random Forest - Acurácia: {accuracy_score(y_clf_test, y_pred_rf):.4f}")

# Regressão
print(f"\nRegressão Linear - R²: {r2_score(y_reg_test, y_pred_lr):.4f}")
print(f"Regressão Linear - RMSE: {np.sqrt(mean_squared_error(y_reg_test, y_pred_lr)):.4f}")
print(f"\nGradient Boosting - R²: {r2_score(y_reg_test, y_pred_gb):.4f}")
print(f"Gradient Boosting - RMSE: {np.sqrt(mean_squared_error(y_reg_test, y_pred_gb)):.4f}")