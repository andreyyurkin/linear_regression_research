# AutoML Regression Library

Библиотека для автоматического машинного обучения в задачах регрессии с автоматической сегментацией данных.

## Структура проекта

```
auto_ml_regression/
├── __init__.py
├── preprocessing/
│   ├── __init__.py
│   └── data_preprocessor.py  # Класс DataPreprocessor
├── segmentation/
│   ├── __init__.py
│   ├── base.py               # Базовый класс сегментации
│   ├── ctb_kmeans.py         # CatBoost + KMeans сегментация
│   ├── kmeans.py             # KMeans сегментация
│   └── gmm.py                # Gaussian Mixture сегментация
├── modeling/
│   ├── __init__.py
│   ├── classifier.py         # Классификатор сегментов
│   └── linear_regs.py        # Линейные регрессоры
├── pipeline.py               # Основной AutoMLPipeline
└── examples/
    └── 01.example.ipynb    # Пример использования
```

## Основные компоненты

### 1. DataPreprocessor
Предобработка данных:
- Заполнение пропусков
- Кодирование категориальных признаков (Target Encoding или OHE)
- Масштабирование числовых признаков (MinMax или Standard)
- Разделение на train/test

### 2. Сегментаторы
- **KMeansSegmenter**: K-means кластеризация по целевому признаку
- **GaussianMixtureSegmenter**: GMM кластеризация
- **CatBoostLeafSegmenter**: Сегментация через листья CatBoost + KMeans

### 3. SegmentClassifier
CatBoost классификатор для предсказания сегментов с подбором гиперпараметров через Optuna

### 4. SegmentLinearModels
Линейные регрессоры для каждого сегмента:
- LinearRegression
- HuberRegressor
- Подбор гиперпараметров через Optuna

### 5. AutoMLPipeline
Объединяет все этапы в единый пайплайн.

## Установка

```bash
git clone https://github.com/yourusername/auto_ml_regression.git
cd auto_ml_regression
pip install -e .
```

## Пример использования

```python
import pandas as pd
from auto_ml_regression import AutoMLPipeline

# Загрузка данных
data = pd.read_csv('insurance.csv')

# Определение признаков
num_features = ['age', 'bmi', 'children']
cat_features = ['sex', 'smoker', 'region']
target_col = 'charges'

# Инициализация пайплайна
pipeline = AutoMLPipeline(
    features=num_features + cat_features,
    cat_features=cat_features,
    target_col=target_col,
    n_segments=3  # Число сегментов
)

# Запуск полного пайплайна
pipeline.preprocess_data(data)
pipeline.segment_data(use_optuna=True, n_trials=30)   # or segment_data_kmeans, or segment_data_gmm
pipeline.train_classifier(n_trials=50)
pipeline.train_regressors(model_type='huber', n_trials=50, cv=4)

# Оценка качества
metrics_df = pipeline.evaluate()
print(metrics_df)

# Доступ к моделям
models = pipeline.get_models()
print("CatBoost модель сегментации:", models['catboost_segmenter'])
print("KMeans модель:", models['kmeans'])
print("Классификатор:", models['classifier'])
print("Регрессоры:", models['regressors'])

# Предсказание на новых данных
new_data = pd.DataFrame(...)  # Новые данные с теми же признаками
preprocessed_data = pipeline.preprocessor.transform(new_data)
segmented_data = pipeline.segmenter.predict(preprocessed_data)
predictions = pipeline.regressor.predict(
    X=segmented_data,
    segments=segmented_data['segment'],
    use_weighted=True,
    segment_probs=pipeline.classifier.model.predict_proba(
        segmented_data[pipeline.features]
    )
)
```

## Метрики качества

Библиотека автоматически вычисляет и возвращает в виде DataFrame на тестовых данных:
- R² (коэффициент детерминации)
- MSE (среднеквадратичная ошибка)
- RMSE (корень из MSE)
- MAE (средняя абсолютная ошибка)
- MAPE (средняя абсолютная процентная ошибка)

## Доступные модели
Через метод `get_models()` можно получить:
1. Модель CatBoost для сегментации
2. Модель KMeans для кластеризации листьев
3. Классификатор сегментов (CatBoost)
4. Словарь регрессоров для каждого сегмента

## Настройки по умолчанию

Все компоненты имеют разумные настройки по умолчанию, но могут быть кастомизированы:
- Тип масштабирования
- Стратегия заполнения пропусков
- Тип кодирования категорий
- Количество сегментов
- Параметры оптимизации (количество trials, CV фолдов)