from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BaseSegmenter(ABC):
    def __init__(self, n_segments=2, plot=False):
        """
        Базовый класс для сегментации данных.
        
        :param n_segments: Количество сегментов
        :param plot: Визуализировать распределения
        """
        self.n_segments = n_segments
        self.plot = plot
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Обучение модели сегментации"""
        pass
    
    def predict(self, X: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Предсказание сегментов для новых данных
        
        :param X: DataFrame с данными
        :param target_col: Название целевой колонки
        :return: DataFrame с добавленной колонкой 'segment'
        """
        if not self._is_fitted:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit()")
            
        data = X.copy()
        segments = self._predict_segments(data[[target_col]])
        data['segment'] = segments
            
        return data
    
    @abstractmethod
    def _predict_segments(self, X_target: pd.DataFrame) -> np.ndarray:
        """Внутренний метод для предсказания сегментов"""
        pass
    
    def _plot_distribution(self, df: pd.DataFrame, target_col: str):
        """Визуализация распределения по сегментам"""
        plt.figure(figsize=(10, 6))
        plt.title(f'Распределение {target_col} по сегментам')
        plt.grid(which='major')
        sns.histplot(data=df, x=target_col, kde=True, hue='segment', bins=25)
        plt.show()
