# Credit Risk Binary Classification

## О проекте
Классификация. Оценка кредитного риска по введённым данным.

## Структура проекта
- credit_risk.ipynb - ноутбук с обучением и настройкой модели
- app.py - веб-интерфейс
- calibrated_clf.pkl - финальная калиброванная модель
- final_xgb_classifier.pkl - финальная некалиброванная модель
- *.pkl - кодировщики признаков

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

<img width="1906" height="878" alt="изображение" src="https://github.com/user-attachments/assets/e7f1aa46-5577-43e9-9b13-714ffb37dd64" />
