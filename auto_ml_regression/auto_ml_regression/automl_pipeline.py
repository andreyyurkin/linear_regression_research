import pandas as pd
import numpy as np
from sklearn import metrics
from typing import Dict, List, Optional, Union

class AutoMLPipeline:
    def __init__(self,
                 num_features: List[str],
                 cat_features: List[str],
                 target_col: str,
                 n_segments: int = 2,
                 test_size: float = 0.33,
                 random_state: int = 2025):
        """
        Полный AutoML пайплайн для регрессии с сегментацией
        
        :param num_features: Список num признаков
        :param cat_features: Список категориальных признаков (исходные названия)
        :param target_col: Название целевой переменной
        :param n_segments: Количество сегментов
        :param random_state: Random seed для воспроизводимости
        """
        self.num_features = num_features
        self.cat_features = cat_features
        self.cat_features_saved = [f'CAT_{col}' for col in cat_features]  # Сохраненные категориальные признаки
        self.target_col = target_col
        self.n_segments = n_segments
        self.test_size = test_size
        self.random_state = random_state
        
        # Инициализация компонентов
        self.preprocessor = None
        self.segmenter = None
        self.classifier = None
        self.regressor = None
        
        # Данные на разных этапах
        self.train_data = None
        self.test_data = None
        self.segmented_train = None
        self.segmented_test = None
    
    def preprocess_data(self, df: pd.DataFrame) -> None:
        """Этап предобработки данных"""
        from auto_ml_regression.preprocessing import DataPreprocessor
        
        self.preprocessor = DataPreprocessor(
            target_col=self.target_col,
            num_features=self.num_features,
            cat_features=self.cat_features,
            cat_encoder_type="target",
            fillna_num_strategy="mean",
            scaler_type="minmax",
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        self.train_data, self.test_data = self.preprocessor.fit_transform(df)
    
    def segment_data(self, use_optuna: bool = True, n_trials: int = 30, 
                     plot: bool = False, depth = 3) -> None:
        """Этап сегментации данных через Catboost+K-means"""
        from auto_ml_regression.segmentation.ctb_kmeans import CatBoostLeafSegmenter
        
        # Для сегментации используем категориальные признаки
        segmenter_features = self.num_features + self.cat_features
        
        self.segmenter = CatBoostLeafSegmenter(
            features=segmenter_features,
            target_col=self.target_col,
            n_segments=self.n_segments,
            depth=depth,
            use_optuna=use_optuna,
            n_trials=n_trials,
            plot=plot,
            #cat_features=self.cat_features_saved  # Передаем сохраненные категориальные признаки
        )
        
        self.segmented_train = self.segmenter.fit(self.train_data)
        self.segmented_test = self.segmenter.predict(self.test_data)

    def segment_data_kmeans(self, plot:bool = False) -> None:
        """Этап сегментации данных через K-means"""
        from auto_ml_regression.segmentation.kmeans import KMeansSegmenter
        self.segmenter = KMeansSegmenter(
            n_segments=self.n_segments,
            plot=plot
        )

        self.segmented_train = self.segmenter.fit(self.train_data, target_col=self.target_col)
        self.segmented_test = self.segmenter.predict(self.test_data, target_col=self.target_col)

    def segment_data_gmm(self, plot:bool = False) -> None:
        """Этап сегментации данных через GMM"""
        from auto_ml_regression.segmentation.gmm import GaussianMixtureSegmenter
        self.segmenter = GaussianMixtureSegmenter(
            n_segments=self.n_segments,
            plot=plot
        )

        self.segmented_train = self.segmenter.fit(self.train_data, target_col=self.target_col)
        self.segmented_test = self.segmenter.predict(self.test_data, target_col=self.target_col)

    
    def train_classifier(self, n_trials: int = 50) -> None:
        """Обучение классификатора сегментов"""
        from auto_ml_regression.modeling.classifier import SegmentClassifier
        
        # Для классификатора используем исходные категориальные признаки
        classifier_features = self.num_features + self.cat_features_saved
        
        self.classifier = SegmentClassifier(
            features=classifier_features,
            cat_features=self.cat_features_saved,
            n_segments=self.n_segments
        )
        
        self.classifier.fit(
            X=self.segmented_train[classifier_features],
            y=self.segmented_train['segment'],
            n_trials=n_trials
        )
        
        # Предсказание на тестовых данных
        if self.n_segments == 2:
            test_segment_probs = self.classifier.model.predict_proba(
                self.segmented_test[classifier_features]
            )[:, 1]
            roc_auc = metrics.roc_auc_score(
                self.segmented_test['segment'],
                test_segment_probs
            )
        else:
            test_segment_probs = self.classifier.model.predict_proba(
                self.segmented_test[classifier_features]
            )
            roc_auc = metrics.roc_auc_score(
                self.segmented_test['segment'],
                test_segment_probs,
                multi_class='ovr'
            )
        
        print(f"TEST ROC AUC = {roc_auc:.4f}")
        
        self.segmented_test['segment_preds'] = self.classifier.model.predict(
            self.segmented_test[classifier_features]
        )
        if self.n_segments == 2:
            self.segmented_test['segment_probs'] = test_segment_probs
    
    def train_regressors(self, 
                        model_type: str = 'huber',
                        use_optuna: bool = True,
                        n_trials: int = 50,
                        cv: int = 4) -> None:
        """Обучение регрессоров для каждого сегмента"""
        from auto_ml_regression.modeling.linear_regs import SegmentLinearModels
        
        # Для регрессоров используем все признаки (закодированные)
        self.regressor = SegmentLinearModels(
            features=self.num_features + self.cat_features,
            model_type=model_type,
            use_optuna=use_optuna,
            n_trials=n_trials,
            cv=cv
        )
        
        self.regressor.fit(
            X=self.segmented_train[self.num_features + self.cat_features],
            y=self.segmented_train[self.target_col],
            segments=self.segmented_train['segment']
        )
    
    # Остальные методы остаются без изменений
    def evaluate_test(self) -> pd.DataFrame:
        """Оценка качества на train и test выборках"""
        test_target_preds = self.regressor.predict(self.segmented_test, 
                              segments=self.segmented_test['segment_preds'],
                              )

        test_target_true = self.segmented_test[self.target_col]

        r2 = metrics.r2_score(test_target_true, test_target_preds)
        mse = metrics.mean_squared_error(test_target_true, test_target_preds)
        mae = metrics.mean_absolute_error(test_target_true, test_target_preds)
        mape = metrics.mean_absolute_percentage_error(test_target_true, test_target_preds)
        metric_names = ['r2', 'mse', 'rmse', 'mae', 'mape']
        metrics_ = [r2, mse, mse**0.5, mae, mape]
        results = pd.Series(metrics_).to_frame().T
        results.columns = metric_names
        return results
    
    def get_models(self) -> Dict[str, object]:
        """Получение всех обученных моделей"""
        return {
            'catboost_segmenter': self.segmenter.cat_model if self.segmenter else None,
            'kmeans': self.segmenter.kmeans if self.segmenter else None,
            'classifier': self.classifier.model if self.classifier else None,
            'regressors': self.regressor.models if self.regressor else None
        }
    
    def get_data(self, stage: str) -> pd.DataFrame:
        """Получение данных на разных этапах обработки
        
        :param stage: Один из ['raw_train', 'raw_test', 'segmented_train', 'segmented_test']
        """
        data_map = {
            'raw_train': self.train_data,
            'raw_test': self.test_data,
            'segmented_train': self.segmented_train,
            'segmented_test': self.segmented_test
        }
        return data_map.get(stage)