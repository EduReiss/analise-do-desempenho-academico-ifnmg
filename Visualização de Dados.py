import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import StringIO

# IMPORTAÇÃO E PRÉ-PROCESSAMENTO 
df = pd.read_csv(r"C:\Users\mateu\Desktop\Programação\Projetos\Projeto multidisciplinar IFNMG\.gitignore\dados.csv")

# Remover linhas sem resultado
df = df[df['Resultado'].isin(['APROVADO', 'REPROVADO'])].copy()

# Converter '-' em NaN nas colunas de nota
nota_cols = ['A_NotaRegular','A_NotaRecuperacao','B_NotaRegular',
             'B_NotaRecuperacao','C_NotaRegular','C_NotaRecuperacao','NotaTotal']
for c in nota_cols:
    df[c] = pd.to_numeric(df[c].replace('-', np.nan), errors='coerce')

# Codificar Resultado como 0/1
df['Resultado_num'] = (df['Resultado'] == 'APROVADO').astype(int)

# Normalizar colunas numéricas (exceto ID e Resultado)
from sklearn.preprocessing import MinMaxScaler
num_cols = df.select_dtypes(include='number').columns.tolist()
scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[num_cols] = scaler.fit_transform(df[num_cols].fillna(0))

print(df.shape, df['Resultado'].value_counts())
# GRÁFICOS DE BARRA: APROVADOS E REPROVADOS POR DISCIPLINA 
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
disciplinas = ['A', 'B', 'C']
cores = ['#1D9E75', '#D85A30']

for i, disc in enumerate(disciplinas):
    col = f'{disc}_NotaRegular'
    ax = axes[i]
    # Usa o resultado final do aluno como proxy
    contagem = df['Resultado'].value_counts()
    barras = ax.bar(contagem.index, contagem.values,
                    color=[cores[0] if r=='APROVADO' else cores[1] for r in contagem.index],
                    edgecolor='none', width=0.5)
    for bar in barras:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=12)
    ax.set_title(f'Disciplina {disc}', fontsize=13)
    ax.set_xlabel('Resultado')
    ax.set_ylabel('Quantidade de Alunos')
    ax.set_ylim(0, max(contagem.values)*1.15)

plt.tight_layout()
plt.savefig('barras_resultado.png', dpi=150)
plt.show()
# BOX PLOTS DAS NOTAS
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
cores_box = ['#1D9E75', '#185FA5', '#D85A30']

for i, disc in enumerate(disciplinas):
    col = f'{disc}_NotaRegular'
    ax = axes[i]
    aprov = df[df['Resultado']=='APROVADO'][col].dropna()
    reprov = df[df['Resultado']=='REPROVADO'][col].dropna()
    bp = ax.boxplot([aprov, reprov], tick_labels=['Aprovados','Reprovados'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set_facecolor(cores_box[i]+'99')
    bp['boxes'][1].set_facecolor('#88888866')
    for element in ['whiskers','caps','fliers']:
        for item in bp[element]:
            item.set_color(cores_box[i])
    ax.set_title(f'Disciplina {disc}', fontsize=13)
    ax.set_ylabel('Nota')
    ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('boxplot_notas.png', dpi=150)
plt.show()
# ENGENHARIA DE FEATURE + VISUALIZAÇÃO DO FÓRUM 
df['Total_Forum'] = (df['Fórum Geral'] + df['Fórum A'] +
                     df['Fórum B'] + df['Fórum C'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: Total_Forum vs NotaTotal
aprov = df[df['Resultado']=='APROVADO']
reprov = df[df['Resultado']=='REPROVADO']
ax = axes[0]
ax.scatter(reprov['Total_Forum'], reprov['NotaTotal'],
           alpha=0.5, color='#D85A30', label='Reprovados', marker='^', s=30)
ax.scatter(aprov['Total_Forum'], aprov['NotaTotal'],
           alpha=0.5, color='#1D9E75', label='Aprovados', marker='o', s=30)
ax.set_xlabel('Total de acessos ao fórum')
ax.set_ylabel('Nota total')
ax.set_title('Fórum vs Nota Total')
ax.legend()

# Barra: média de acessos ao fórum
medias = df.groupby('Resultado')['Total_Forum'].mean().reindex(['APROVADO','REPROVADO'])
ax2 = axes[1]
bars = ax2.bar(medias.index, medias.values, color=['#1D9E75','#D85A30'],
               edgecolor='none', width=0.5)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=12)
ax2.set_title('Média de acessos ao fórum por resultado')
ax2.set_ylabel('Média de acessos ao fórum')
ax2.set_ylim(0, medias.max()*1.2)

plt.tight_layout()
plt.savefig('forum_desempenho.png', dpi=150)
plt.show()

print("Média aprovados:", round(medias['APROVADO'], 2))
print("Média reprovados:", round(medias['REPROVADO'], 2))