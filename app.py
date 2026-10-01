import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


@st.cache_resource
def load_all():
    model = keras.models.load_model('model_nn.keras')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    means = pd.read_csv('feature_stats.csv', index_col=0)['mean']
    return model, scaler, features, means


model, scaler, features, means = load_all()
low = dict(zip(features, scaler.data_min_))
high = dict(zip(features, scaler.data_max_))

st.title('Рекомендация соотношения матрица-наполнитель')
st.write('Введите параметры компонентов композита. По умолчанию подставлены средние значения из датасета, можно ввести любые.')

values = []
out_of_range = []
for name in features:
    default = 0.0 if name == 'Угол нашивки, град' else round(float(means[name]), 3)
    v = st.number_input(name, value=default, format='%.3f')
    if v < low[name] or v > high[name]:
        st.warning(f'Значение вне диапазона обучающей выборки: от {low[name]:.3f} до {high[name]:.3f}. '
                   'Прогноз может быть ненадёжен.')
        out_of_range.append(name)
    values.append(v)

if st.button('Рассчитать'):
    x = scaler.transform(np.array(values).reshape(1, -1))
    y = float(model.predict(x, verbose=0)[0][0])
    st.success(f'Рекомендуемое соотношение матрица-наполнитель: {y:.3f}')
    if out_of_range:
        st.warning(f'Вне диапазона обучающей выборки: {len(out_of_range)} из {len(features)} признаков. '
                   'Модель не обучалась на таких данных, прогноз может быть ненадёжен.')
