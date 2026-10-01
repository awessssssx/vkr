import os
import pandas as pd

os.makedirs('output', exist_ok=True)

X_bp = pd.read_excel('Датасет для ВКР_композиты/X_bp.xlsx', index_col=0)
X_nup = pd.read_excel('Датасет для ВКР_композиты/X_nup.xlsx', index_col=0)
print('X_bp: ', X_bp.shape)
print('X_nup:', X_nup.shape)

df = X_bp.join(X_nup, how='inner')
print('После объединения:', df.shape)

print('\nПропуски по колонкам:')
print(df.isna().sum())

stats = pd.DataFrame({
    'Среднее': df.mean(),
    'Медиана': df.median(),
    'Минимум': df.min(),
    'Максимум': df.max(),
}).round(3)
print('\nОписательные статистики:')
print(stats)
stats.to_csv('output/table_stats.csv')

df.to_csv('data.csv')
print('\nСохранено: data.csv, output/table_stats.csv')
