import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv', index_col=0)

df.hist(figsize=(20, 15), bins=30)
plt.suptitle('Гистограммы распределения переменных', fontsize=16)
plt.tight_layout()
plt.savefig('output/01_histograms.png', dpi=150)
plt.close()

fig, axes = plt.subplots(4, 4, figsize=(20, 15))
for ax, col in zip(axes.flatten(), df.columns):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col, fontsize=10)
    ax.set_ylabel('')
for ax in axes.flatten()[len(df.columns):]:
    ax.axis('off')
plt.suptitle('Диаграммы «ящик с усами»', fontsize=16)
plt.tight_layout()
plt.savefig('output/02_boxplots.png', dpi=150)
plt.close()

g = sns.pairplot(df, plot_kws={'s': 5})
g.fig.suptitle('Попарные графики рассеяния', y=1.01, fontsize=16)
plt.savefig('output/03_pairplot.png', dpi=80)
plt.close()

plt.figure(figsize=(14, 12))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Матрица корреляций', fontsize=16)
plt.tight_layout()
plt.savefig('output/04_corr_heatmap.png', dpi=150)
plt.close()

print('Сохранено: output/01_histograms.png, 02_boxplots.png, 03_pairplot.png, 04_corr_heatmap.png')
