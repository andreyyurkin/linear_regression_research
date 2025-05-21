import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import cross_val_score
import optuna
from typing import Dict, Union, Tuple

class SegmentLinearModels:
    def __init__(self, 
                 features: list,
                 model_type: str = 'linear',
                 use_optuna: bool = False,
                 n_trials: int = 20,
                 cv: int = 5):
        """
        Класс для построения линейных моделей на сегментах данных.
        
        :param features: Список признаков для регрессии
        :param model_type: Тип модели ('linear' или 'huber')
        :param use_optuna: Использовать Optuna для подбора гиперпараметров
        :param n_trials: Количество итераций Optuna
        :param cv: Количество фолдов для кросс-валидации
        """
        self.features = features
        self.model_type = model_type
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.cv = cv
        self.models = {}  
        self.best_params = {}  
        self.segment_weights = {} 
        
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            segments: pd.Series) -> 'SegmentLinearModels':
        """
        Обучение моделей для каждого сегмента.
        
        :param X: Признаки
        :param y: Целевая переменная
        :param segments: Сегменты (метки классов)
        :return: self
        """
        unique_segments = segments.unique()
        
        for seg in unique_segments:
            mask = segments == seg
            X_seg = X[mask][self.features]
            y_seg = y[mask]
            
            if self.use_optuna:
                self._optimize_and_fit(seg, X_seg, y_seg)
            else:
                self._simple_fit(seg, X_seg, y_seg)
            
            # Сохраняем вес сегмента (долю наблюдений)
            self.segment_weights[seg] = len(X_seg) / len(X)
            
        return self
    
    def _simple_fit(self, 
                   seg: int, 
                   X: pd.DataFrame, 
                   y: pd.Series) -> None:
        """Простое обучение без подбора гиперпараметров"""
        if self.model_type == 'huber':
            model = HuberRegressor()
        else:
            model = LinearRegression()
            
        model.fit(X, y)
        self.models[seg] = model
        self.best_params[seg] = model.get_params()
    
    def _optimize_and_fit(self, 
                         seg: int, 
                         X: pd.DataFrame, 
                         y: pd.Series) -> None:
        """Подбор гиперпараметров и обучение с Optuna"""
        def objective(trial):
            if self.model_type == 'huber':
                params = {
                    'epsilon': trial.suggest_float('epsilon', 1.0, 10.0),
                    'alpha': trial.suggest_float('alpha', 0.0001, 0.3),
                    'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
                }
                model = HuberRegressor(**params, max_iter = 2_000)
            else:
                params = {
                    'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                    'positive': trial.suggest_categorical('positive', [True, False])
                }
                model = LinearRegression(**params)
            
            score_rmse = cross_val_score(
                model, X, y, 
                cv=self.cv,
                scoring='neg_root_mean_squared_error'
            ).mean()

            score_mae = cross_val_score(
                model, X, y, 
                cv=self.cv,
                scoring='neg_mean_absolute_error'
            ).mean()
            
            return (score_rmse+score_mae)/2
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        if self.model_type == 'huber':
            self.models[seg] = HuberRegressor(**study.best_params, max_iter = 2_000)
        else:
            self.models[seg] = LinearRegression(**study.best_params)
            
        self.models[seg].fit(X, y)
        self.best_params[seg] = study.best_params
    
    def predict(self, 
            X: pd.DataFrame, 
            segments: pd.Series,
            use_weighted: bool = False,
            segment_probs: np.ndarray = None) -> np.ndarray:
          """
          Предсказание для каждого наблюдения по его сегменту.
          
          :param X: Признаки
          :param segments: Предсказанные сегменты
          :param use_weighted: Использовать взвешенные предсказания
          :param segment_probs: Вероятности принадлежности к сегментам (n_samples, n_segments)
          :return: Массив предсказаний
          """
          if use_weighted and segment_probs is None:
               raise ValueError("Для взвешенных предсказаний нужны segment_probs")
               
          predictions = np.zeros(len(X))
          
          if use_weighted:
               # Определяем маски для уверенных и сомневающихся наблюдений
               if len(segments.unique()) == 2:
                    mask_0 = segment_probs[:, 1] <= 0.4
                    mask_1 = segment_probs[:, 1] >= 0.6
                    mask_doubting = ~mask_1 & ~mask_0
               else:
                   mask_0 = segment_probs <= 0.4
                   mask_1 = segment_probs >= 0.6
                   mask_doubting = ~mask_0 & ~mask_1
               
               # Обрабатываем каждый сегмент
               for seg in self.models.keys():
                    mask_seg = segments == seg
                    if not any(mask_seg):
                         continue
                         
                    # Получаем предсказания для всего сегмента
                    pred = self.models[seg].predict(X[mask_seg][self.features])
                    
                    # Уверенные предсказания
                    if len(segments.unique()) == 2:
                         confident_mask = mask_seg & ~mask_doubting[:]
                    else:
                        confident_mask = mask_seg & ~mask_doubting[:, seg]

                    predictions[confident_mask] = pred[confident_mask[mask_seg]]
                    
                    # Сомневающиеся предсказания (только если есть такие наблюдения)
                    if len(segments.unique()) == 2:
                         doubting_mask = mask_seg & mask_doubting[:]
                    else:
                        doubting_mask = mask_seg & mask_doubting[:, seg]

                    if any(doubting_mask):
                         # Берем только нужные элементы из pred и segment_probs
                         doubting_idx = doubting_mask[mask_seg]
                         prob_idx = 1 if seg == 1 else 0  # Вероятность текущего сегмента
                         predictions[doubting_mask] = (
                              pred[doubting_idx] * 
                              segment_probs[doubting_mask, prob_idx]
                         )
          else:
               # Стандартные предсказания
               for seg in self.models.keys():
                    mask = segments == seg
                    if any(mask):
                         predictions[mask] = self.models[seg].predict(X[mask][self.features])
          
          return predictions
    
    def get_model(self, segment: int) -> Union[LinearRegression, HuberRegressor]:
        """Получить модель для конкретного сегмента"""
        return self.models.get(segment)
    
    def get_params(self, segment: int) -> dict:
        """Получить параметры модели для сегмента"""
        return self.best_params.get(segment, {})
    
    def get_model_coefs(self) -> pd.DataFrame:
        N = len(self.models)
        stat = pd.DataFrame(columns = [f'model_coef_segment_{m}' for m in range(N)],
                    index = ['intercept_'] + list(self.models[0].feature_names_in_))
        for m in range(N):
            x = list([self.models[m].intercept_])
            x.extend(list(self.models[m].coef_))
            stat[f'model_coef_segment_{m}'] = x

        return stat