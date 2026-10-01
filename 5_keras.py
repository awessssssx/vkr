import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)

df = pd.read_csv('data_clean.csv', index_col=0)
TARGET = 'Соотношение матрица-наполнитель'

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(X_train_s, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Обучающая выборка')
plt.plot(history.history['val_loss'], label='Валидационная выборка')
plt.xlabel('Эпоха')
plt.ylabel('Ошибка (MSE)')
plt.title('Кривая обучения нейронной сети')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/07_nn_training.png', dpi=150)
plt.close()

pred_train = model.predict(X_train_s, verbose=0).ravel()
pred_test = model.predict(X_test_s, verbose=0).ravel()
results = pd.DataFrame({
    'MAE': [mean_absolute_error(y_train, pred_train), mean_absolute_error(y_test, pred_test)],
    'R2': [r2_score(y_train, pred_train), r2_score(y_test, pred_test)],
}, index=['Обучающая выборка', 'Тестовая выборка']).round(3)
print(results)
results.to_csv('output/table_results_nn.csv')

model.save('model_nn.keras')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'features.pkl')
X.mean().rename('mean').to_csv('feature_stats.csv')  # средние значения для полей приложения
print('\nСохранено: model_nn.keras, scaler.pkl, features.pkl, feature_stats.csv, output/07_nn_training.png, table_results_nn.csv')
