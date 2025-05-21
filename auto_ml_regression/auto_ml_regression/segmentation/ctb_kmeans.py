from abc import ABC
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from .base import BaseSegmenter
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Union
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

class CatBoostLeafSegmenter(BaseSegmenter):
    def __init__(self,
                 features: List[str],
                 target_col: str,
                 n_segments: int = 2,
                 plot: bool = False,
                 cat_features: Optional[List[str]] = None,
                 depth: int = 3,
                 use_optuna: bool = False,
                 n_trials: int = 20):
        """
        Сегментация через CatBoost leaves + KMeans
        
        :param features: Список признаков для использования в CatBoost
        :param target_col: Название целевой переменной
        :param n_segments: Количество финальных кластеров (сегментов)
        :param plot: Визуализировать распределения
        :param cat_features: Список категориальных признаков
        :param depth: Глубина деревьев CatBoost
        :param use_optuna: Использовать Optuna для подбора гиперпараметров
        :param n_trials: Количество итераций Optuna
        """
        super().__init__(n_segments=n_segments, plot=plot)
        self.features = features
        self.target_col = target_col
        self.cat_features = cat_features or []
        self.depth = depth
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.cat_model = None
        self.kmeans = None
        self.scaler = MinMaxScaler()
        self.leaf_to_cluster_map = {}
        
    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Обучение модели сегментации
        
        :param X: DataFrame с данными (должен содержать self.features и self.target_col)
        :return: DataFrame с добавленной колонкой 'segment'
        """
        self._validate_features(X)
        data = X.copy()
        y = data[self.target_col]
        
        # 1. Обучение CatBoost
        self._fit_catboost(data[self.features], y)
        
        # 2. Получение индексов листьев
        train_leaf_indices = self._get_leaf_indices(data[self.features])
        data['leaf'] = train_leaf_indices
        
        # 3. Сбор статистики по листьям
        leaf_stats = self._get_leaf_stats(train_leaf_indices, data[self.features], y)
        
        # 4. Кластеризация листьев
        cluster_features = self._get_cluster_features(leaf_stats)
        scaled_stats = self.scaler.fit_transform(leaf_stats[cluster_features])
        
        self.kmeans = KMeans(n_clusters=self.n_segments, random_state=2025)
        leaf_stats['cluster'] = self.kmeans.fit_predict(scaled_stats)
        
        # 5. Сопоставление листьев и кластеров
        self.leaf_to_cluster_map = leaf_stats.set_index('leaf')['cluster'].to_dict()
        data['segment'] = data['leaf'].map(self.leaf_to_cluster_map)
        
        if self.plot:
            self._plot_distribution(data, self.target_col)
            self._plot_cluster_stats(leaf_stats)
        
        self._is_fitted = True
        return data
    
    def _validate_features(self, X: pd.DataFrame):
        """Проверка наличия всех необходимых признаков"""
        missing_features = set(self.features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Отсутствуют признаки: {missing_features}")
            
        if self.target_col not in X.columns:
            raise ValueError(f"Отсутствует целевая переменная: {self.target_col}")
    
    def _fit_catboost(self, X: pd.DataFrame, y: pd.Series):
     """Обучение CatBoost модели с подбором гиперпараметров"""
     if self.use_optuna:
          study = optuna.create_study(direction='maximize')  # Используем maximize для neg_rmse
          study.optimize(lambda trial: self._objective(trial, X, y), 
                         n_trials=self.n_trials)
          
          self.cat_model = CatBoostRegressor(
               iterations=1,
               depth=study.best_params['depth'],
               learning_rate=study.best_params['learning_rate'],
               l2_leaf_reg=study.best_params['l2_leaf_reg'],
               loss_function='RMSE',  # Явно указываем функцию потерь
               cat_features=self.cat_features,
               verbose=0
          )
     else:
          self.cat_model = CatBoostRegressor(
               iterations=1,
               depth=self.depth,
               loss_function='RMSE',  # Явно указываем функцию потерь
               cat_features=self.cat_features,
               verbose=0
          )
     
     self.cat_model.fit(X, y)
    
    def _objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
     """Функция для Optuna с явным указанием loss_function"""
     params = {
          'depth': trial.suggest_int('depth', 2, 4),
          'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
          'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
          'iterations': 1,
          'loss_function': 'RMSE',  # Явно указываем функцию потерь
          'cat_features': self.cat_features,
          'verbose': False
     }
     
     model = CatBoostRegressor(**params)
     
     # Используем встроенную кросс-валидацию sklearn
     scores = cross_val_score(
          model,
          X,
          y,
          cv=3,
          scoring='neg_root_mean_squared_error',
          n_jobs=-1
     )
     
     return scores.mean()
    
    def _get_leaf_indices(self, X: pd.DataFrame) -> np.ndarray:
        """Получение индексов листьев"""
        pool = Pool(X, cat_features=self.cat_features)
        return self.cat_model.calc_leaf_indexes(pool).flatten()
    
    def _get_leaf_stats(self, 
                       leaf_indices: np.ndarray, 
                       X: pd.DataFrame, 
                       y: pd.Series) -> pd.DataFrame:
        """Сбор статистики по листьям"""
        stats_list = []
        for leaf in np.unique(leaf_indices):
            mask = leaf_indices == leaf
            stat = {
                'leaf': leaf,
                'n_samples': np.sum(mask),
                'target_mean': np.mean(y[mask]),
                'target_median': np.median(y[mask]),
                'target_std': np.std(y[mask]),
                'target_min': np.min(y[mask]),
                'target_max': np.max(y[mask])
            }
            
            # Статистики по признакам
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    stat[f'{col}_mean'] = np.mean(X.loc[mask, col])
                    stat[f'{col}_median'] = np.median(X.loc[mask, col])
                    stat[f'{col}_std'] = np.std(X.loc[mask, col])
                else:
                    # Для категориальных - мода
                    stat[f'{col}_mode'] = X.loc[mask, col].mode()[0]
            
            stats_list.append(stat)
        
        return pd.DataFrame(stats_list)
    
    def _get_cluster_features(self, leaf_stats: pd.DataFrame) -> List[str]:
        """Выбор признаков для кластеризации"""
        exclude = ['leaf', 'n_samples']
        return [col for col in leaf_stats.columns 
               if col not in exclude and not col.startswith(('target_'))]
    
    def _plot_cluster_stats(self, leaf_stats: pd.DataFrame):
        """Визуализация статистик кластеров"""
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            leaf_stats.groupby('cluster').mean().T,
            annot=True, cmap='coolwarm'
        )
        plt.title('Средние значения признаков по кластерам')
        plt.show()
    
    def _predict_segments(self, X_target: pd.DataFrame) -> np.ndarray:
        """Для совместимости с BaseSegmenter"""
        raise NotImplementedError("Используйте predict() вместо прямого вызова этого метода")
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Предсказание сегментов для новых данных
        
        :param X: DataFrame с данными (должен содержать self.features)
        :return: DataFrame с добавленной колонкой 'segment'
        """
        if not self._is_fitted:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit()")
            
        self._validate_features(X)
        data = X.copy()
        
        # Получаем листья
        leaf_indices = self._get_leaf_indices(data[self.features])
        data['leaf'] = leaf_indices
        
        # Сопоставляем с кластерами
        data['segment'] = data['leaf'].map(
            lambda x: self.leaf_to_cluster_map.get(x, -1)  # -1 для новых листьев
        )
        
        return data