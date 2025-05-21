import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from .base import BaseSegmenter
import matplotlib.pyplot as plt
import seaborn as sns

class GaussianMixtureSegmenter(BaseSegmenter):
    def __init__(self, n_segments=2, plot=False, 
                 covariance_type='full', random_state=2025):
        """
        :param covariance_type: Тип ковариационной матрицы ('full', 'tied', 'diag', 'spherical')
        :param random_state: Для воспроизводимости результатов
        """
        super().__init__(n_segments, plot)
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = None
        self.means_ = None
        self.covariances_ = None
    
    def fit(self, X: pd.DataFrame, target_col: str) -> pd.DataFrame:
        # Создаем копию данных для безопасности
        data = X.copy()
        
        # Инициализируем и обучаем GMM
        self.gmm = GaussianMixture(
            n_components=self.n_segments,
            covariance_type=self.covariance_type,
            random_state=self.random_state
        )
        self.gmm.fit(data[[target_col]])
        
        # Получаем параметры распределений
        self.means_ = self.gmm.means_.flatten()
        self.covariances_ = self.gmm.covariances_
        
        # Предсказываем сегменты
        segments = self.gmm.predict(data[[target_col]])
        data['segment'] = segments
        
        # Визуализация если требуется
        if self.plot:
            self._plot_distribution(data, target_col)
            self._plot_density(data, target_col)

        self._is_fitted = True

        return data
    
    def _plot_density(self, df: pd.DataFrame, target_col: str):
        """Дополнительный график с плотностями распределений"""
        plt.figure(figsize=(12, 6))
        
        # Гистограмма данных
        sns.histplot(data=df, x=target_col, hue='segment', 
                    bins=30, kde=False, stat='density', 
                    common_norm=False, alpha=0.5)
        
        # Отрисовка плотностей GMM
        x = np.linspace(df[target_col].min(), df[target_col].max(), 500)
        for i in range(self.n_segments):
            y = (self.gmm.weights_[i] * 
                 self._gaussian(x, self.means_[i], np.sqrt(self.covariances_[i][0][0])))
            plt.plot(x, y, label=f'Segment {i} density', linewidth=2)
        
        plt.title('GMM Density Estimation by Segment')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def _gaussian(x, mu, sigma):
        """Функция плотности нормального распределения"""
        return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    
    def get_segment_stats(self) -> dict:
        """Возвращает статистику по сегментам"""
        if not hasattr(self, 'means_'):
            raise RuntimeError("Model not fitted yet")
        
        stats = {}
        for i in range(self.n_segments):
            stats[f'segment_{i}'] = {
                'mean': self.means_[i],
                'covariance': self.covariances_[i][0][0],
                'weight': self.gmm.weights_[i]
            }
        return stats
    
    def _predict_segments(self, X_target: pd.DataFrame) -> np.ndarray:
        return self.gmm.predict(X_target)