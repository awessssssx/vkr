import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data.csv', index_col=0)


def is_outlier(s):
    """Выброс - значение вне интервала [Q1 - 1.5*IQR; Q3 + 1.5*IQR]"""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


cols = [c for c in df.columns if c != 'Угол нашивки, град']

outliers = pd.Series({c: int(is_outlier(df[c]).sum()) for c in cols}, name='Число выбросов')
print('Число выбросов по колонкам:')
print(outliers)
outliers.to_csv('output/table_outliers.csv')

mask = pd.Series(False, index=df.index)
for c in cols:
    mask |= is_outlier(df[c])
df_clean = df[~mask]
print('\nСтрок до удаления выбросов:', len(df))
print('Удалено строк:', int(mask.sum()))
print('Осталось строк:', len(df_clean))
df_clean.to_csv('data_clean.csv')

scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns, index=df_clean.index)

df_clean.hist(figsize=(20, 15), bins=30)
plt.suptitle('Гистограммы до нормализации', fontsize=16)
plt.tight_layout()
plt.savefig('output/05_hist_before_norm.png', dpi=150)
plt.close()

df_norm.hist(figsize=(20, 15), bins=30)
plt.suptitle('Гистограммы после нормализации', fontsize=16)
plt.tight_layout()
plt.savefig('output/06_hist_after_norm.png', dpi=150)
plt.close()

minmax = pd.DataFrame({
    'Минимум до': df_clean.min(),
    'Максимум до': df_clean.max(),
    'Минимум после': df_norm.min(),
    'Максимум после': df_norm.max(),
}).round(3)
print('\nМинимумы и максимумы до и после нормализации:')
print(minmax)
minmax.to_csv('output/table_minmax.csv')
print('\nСохранено: data_clean.csv, output/05_hist_before_norm.png, 06_hist_after_norm.png, table_outliers.csv, table_minmax.csv')
