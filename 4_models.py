import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('data_clean.csv', index_col=0)

TARGET_1 = 'Модуль упругости при растяжении, ГПа'
TARGET_2 = 'Прочность при растяжении, МПа'

models = {
    'Линейная регрессия': (LinearRegression(), {'fit_intercept': [True, False]}),
    'K ближайших соседей': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 10, 20, 50]}),
    'Случайный лес': (RandomForestRegressor(random_state=42), {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}),
}

for target, fname in [(TARGET_1, 'modulus'), (TARGET_2, 'strength')]:
    print('\n=== Цель:', target, '===')
    X = df.drop(columns=[TARGET_1, TARGET_2])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    rows = []
    for name, (model, params) in models.items():
        gs = GridSearchCV(model, params, cv=10, scoring='neg_mean_absolute_error')
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        rows.append({
            'Модель': name,
            'Лучшие параметры': str(gs.best_params_),
            'MAE train': mean_absolute_error(y_train, best.predict(X_train)),
            'MAE test': mean_absolute_error(y_test, best.predict(X_test)),
            'R2 train': r2_score(y_train, best.predict(X_train)),
            'R2 test': r2_score(y_test, best.predict(X_test)),
        })
        print(f'{name}: лучшие параметры {gs.best_params_}, MAE test = {rows[-1]["MAE test"]:.3f}')

    results = pd.DataFrame(rows).set_index('Модель').round(3)
    print(results)
    results.to_csv(f'output/table_results_{fname}.csv')

print('\nСохранено: output/table_results_modulus.csv, output/table_results_strength.csv')
