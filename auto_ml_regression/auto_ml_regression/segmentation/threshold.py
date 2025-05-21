import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from .base import BaseSegmenter
import matplotlib.pyplot as plt
import seaborn as sns

class ThresholdSegmenter(BaseSegmenter):
    def __init__(self, 
                 thresholds=None, 
                 n_segments=None,
                 auto_detect=False,
                 plot=False):
        """
        :param thresholds: Список пороговых значений (например [100, 200])
        :param n_segments: Число сегментов (используется при auto_detect=True)
        :param auto_detect: Автоматически определять пороги через GMM
        :param plot: Визуализировать распределение
        """
        super().__init__(n_segments=n_segments or (len(thresholds)+1 if thresholds else 2), 
                        plot=plot)
        self.thresholds = sorted(thresholds) if thresholds else None
        self.auto_detect = auto_detect
        self.gmm = None
        
    def fit(self, X: pd.DataFrame, target_col: str) -> pd.DataFrame:
        data = X.copy()
        
        if self.auto_detect and not self.thresholds:
            self._auto_detect_thresholds(data[target_col])
        
        # Применяем пороговую сегментацию
        data['segment'] = 0
        if self.thresholds:
            for i, threshold in enumerate(self.thresholds):
                data.loc[data[target_col] > threshold, 'segment'] = i + 1
        
        if self.plot:
            self._plot_distribution(data, target_col)
            if self.auto_detect:
                self._plot_thresholds(data, target_col)
     
        self._is_fitted = True
        
        return data
    
    def _auto_detect_thresholds(self, values: pd.Series):
        """Автоматическое определение порогов через GMM"""
        self.gmm = GaussianMixture(n_components=self.n_segments)
        self.gmm.fit(values.values.reshape(-1, 1))
        
        # Получаем средние и сортируем
        means = self.gmm.means_.flatten()
        sorted_means = np.sort(means)
        
        # Вычисляем пороги как середины между средними
        self.thresholds = [
            (sorted_means[i] + sorted_means[i+1]) / 2 
            for i in range(len(sorted_means)-1)
        ]
    
    def _plot_thresholds(self, df: pd.DataFrame, target_col: str):
        """Визуализация порогов на распределении"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=target_col, bins=30)
        
        for i, threshold in enumerate(self.thresholds):
            plt.axvline(threshold, color='r', linestyle='--', 
                        label=f'Threshold {i+1}: {threshold:.2f}')
        
        plt.title('Automatic Threshold Detection')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_thresholds(self) -> list:
        """Возвращает используемые пороговые значения"""
        if self.thresholds is None:
            raise RuntimeError("Thresholds not determined yet. Call fit() first.")
        return self.thresholds
    
    def _predict_segments(self, X_target: pd.DataFrame) -> np.ndarray:
          segments = np.zeros(len(X_target))
          if self.thresholds:
               for i, threshold in enumerate(self.thresholds):
                    segments[X_target.iloc[:, 0] > threshold] = i + 1
          return segments