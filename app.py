import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Настройка страницы
st.set_page_config(page_title="Система кредитного скоринга", layout="wide")

st.title("🏦 Система прогнозирования кредитного риска")
st.markdown("---")

# Загрузка модели
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, 'calibrated_clf.pkl'))
    le_grade = joblib.load(os.path.join(BASE_DIR, 'le_grade.pkl'))
    le_default = joblib.load(os.path.join(BASE_DIR, 'le_default.pkl'))
    ohe = joblib.load(os.path.join(BASE_DIR, 'ohe.pkl'))
    return model, le_grade, le_default, ohe

try:
    model, le_grade, le_default, ohe = load_model()
except:
    st.error("Не удалось загрузить модель. Убедитесь, что файлы сохранены.")
    st.stop()

# Боковая панель с вводными данными
st.sidebar.header("📋 Параметры заёмщика")

person_age = st.sidebar.number_input("Возраст", min_value=18, max_value=100, value=28)
person_income = st.sidebar.number_input("Годовой доход", min_value=0, value=56000)
person_home_ownership = st.sidebar.selectbox("Владение жильём", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.sidebar.number_input("Трудовой стаж (лет)", min_value=0, max_value=60, value=12)
loan_intent = st.sidebar.selectbox("Цель кредита", ["PERSONAL", "MEDICAL", "EDUCATION", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.sidebar.selectbox("Кредитный рейтинг", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Сумма кредита", min_value=1000, value=5525)
loan_int_rate = st.sidebar.number_input("Процентная ставка", min_value=0.0, max_value=30.0, value=12.68)
loan_percent_income = st.sidebar.number_input("Долговая нагрузка", min_value=0.0, max_value=2.0, value=0.10)
cb_person_default_on_file = st.sidebar.selectbox("Дефолты в прошлом", ["N", "Y"])
cb_person_cred_hist_length = st.sidebar.number_input("Длина кредитной истории (лет)", min_value=0, value=9)

# Кнопка прогноза
if st.sidebar.button("🔮 Рассчитать риск", type="primary"):
    # Подготовка данных
    client_data = {
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    }
    
    client_df = pd.DataFrame(client_data)
    
    # Кодирование
    client_df['loan_grade_encoded'] = le_grade.transform(client_df['loan_grade'])
    client_df['cb_person_default_on_file_encoded'] = le_default.transform(client_df['cb_person_default_on_file'])
    client_df = client_df.drop(columns=['loan_grade', 'cb_person_default_on_file'])
    
    # One-Hot Encoding
    ohe_features = ohe.transform(client_df[['person_home_ownership', 'loan_intent']])
    ohe_columns = ohe.get_feature_names_out(['person_home_ownership', 'loan_intent'])
    ohe_df = pd.DataFrame(ohe_features, columns=ohe_columns, index=client_df.index)
    
    client_df = pd.concat([client_df, ohe_df], axis=1)
    client_df = client_df.drop(columns=['person_home_ownership', 'loan_intent'])
    
    # Выравнивание колонок
    client_processed = client_df[model.feature_names_in_]
    
    # Прогноз
    prob_defolt = model.predict_proba(client_processed)[0][1]
    prob_percent = prob_defolt * 100
    
    # Отображение результата
    st.header("📊 Результат прогнозирования")
    
    # Цветовая индикация риска
    if prob_percent < 10:
        risk_level = "🟢 Низкий риск"
        color = "green"
        recommendation = "✅ Рекомендуется одобрение на стандартных условиях"
    elif prob_percent < 30:
        risk_level = "🟡 Умеренный риск"
        color = "orange"
        recommendation = "⚠️ Условное одобрение с корректировкой условий"
    elif prob_percent < 60:
        risk_level = "🟠 Повышенный риск"
        color = "red"
        recommendation = "❌ Рекомендуется отказ или серьёзное повышение ставки"
    else:
        risk_level = "🔴 Критический риск"
        color = "darkred"
        recommendation = "❌ Не рекомендуется одобрение"
    
    # Метрики
    col1, col2, col3 = st.columns(3)
    col1.metric("Вероятность дефолта", f"{prob_percent:.2f}%")
    col2.metric("Вероятность погашения", f"{100 - prob_percent:.2f}%")
    col3.markdown(f"### {risk_level}")
    
    st.info(recommendation)
    
    # SHAP анализ
    st.markdown("---")
    st.header("🔍 Объяснение прогноза (SHAP)")
    
    explainer = shap.Explainer(model.estimator)
    shap_values = explainer(client_processed)
    
    # Водопадная диаграмма
    st.subheader("Влияние факторов на прогноз")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
    
    # Таблица с факторами
    st.subheader("Топ-10 факторов риска")
    shap_df = pd.DataFrame({
        'Признак': client_processed.columns,
        'SHAP_вклад': shap_values.values[0],
        'Значение': client_processed.iloc[0].values
    }).sort_values(by='SHAP_вклад', ascending=False).head(10)
    
    # Добавляем цвета
    shap_df['Влияние'] = shap_df['SHAP_вклад'].apply(lambda x: '⬆️ Повышает риск' if x > 0 else '⬇️ Снижает риск')
    st.dataframe(shap_df, use_container_width=True)
    
    # Рекомендации
    st.markdown("---")
    st.header("💡 Рекомендации")
    
    if prob_percent >= 10:
        st.markdown("""
        **Предлагаемые условия:**
        - Снизить сумму кредита на 20-30%
        - Повысить процентную ставку на 2-3%
        - Запросить поручителя или залог
        - Установить ежемесячный мониторинг
        """)
    else:
        st.markdown("""
        **Предлагаемые условия:**
        - Одобрить запрошенную сумму
        - Стандартная процентная ставка
        - Базовый мониторинг
        """)

# Информация о системе
st.markdown("---")
st.markdown("""
### ℹ️ О системе
Система основана на модели **XGBoost** с байесовской оптимизацией гиперпараметров 
и калибровкой вероятностей. Для интерпретации решений используется метод **SHAP**.

**Метрики качества:**
- ROC-AUC: 0.9511
- Brier Score: 0.0544
- Время прогноза: < 1 секунды
""")
