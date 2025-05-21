from sklearn.cluster import KMeans
from .base import BaseSegmenter
import pandas as pd
import numpy as np

class KMeansSegmenter(BaseSegmenter):
    def fit(self, X: pd.DataFrame, target_col: str):
        kmeans = KMeans(n_clusters=self.n_segments, random_state=2025)
        segments = kmeans.fit_predict(X[[target_col]])
        
        X['segment'] = segments
        
        if self.plot:
            self._plot_distribution(X, target_col)

        self._is_fitted = True    
        self.kmeans = kmeans
        return X
    
    def _predict_segments(self, X_target: pd.DataFrame) -> np.ndarray:
        return self.kmeans.predict(X_target)