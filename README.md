# Credit Risk Binary Classification

Классификация. Оценка кредитного риска по введённым данным.

Произведено сравнение 8 ансамблевых алгоритмов машинного обучения. Финальным выбран XGBoost.

Оптимизирована метрика ROC-AUC c помощью библиотеки Optuna.

Platt Scalling калибровка вероятностей.

SHAP интерпретация.

Создан интерфейс на Streamlit.

## Метрики качества
- ROC-AUC: 0.9489
- Brier Score: 0.0544

Dataset - https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data

## Использование

Streamlit App - https://credit-risk-binary-classification-ckdrnazwapjafczxkzwjd2.streamlit.app/

1. Откройте streamlit app
2. Введите параметры заёмщика в боковой панели
3. Нажмите "Рассчитать риск"
4. Получите прогноз вероятности дефолта и рекомендации


