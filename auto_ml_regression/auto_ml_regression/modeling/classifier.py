import optuna
import catboost as cat
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

class SegmentClassifier:
    def __init__(self, features: list, cat_features: list = None, n_segments:int = 2):
        self.features = features
        self.cat_features = cat_features or []
        self.n_segments = n_segments
        self.model = None
        self.best_params = None
    
    def _objective(self, trial, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2025, stratify=y)

        # Параметры для перебора
        params = {
               'iterations': trial.suggest_int('iterations', 100, 2000),
               'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
               'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
               'random_strength': trial.suggest_float('random_strength', 1e-3, 10, log=True),
               'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
               'border_count': trial.suggest_int('border_count', 32, 255),
               'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
               'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),
               'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
               'boosting_type': 'Plain',  # Оставим только Plain для стабильности
               'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
               'od_wait': trial.suggest_int('od_wait', 10, 50),
               'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 255),
          }
        if params['grow_policy'] == 'Lossguide':
          params['max_leaves'] = trial.suggest_int('max_leaves', 4, 64)
        elif params['grow_policy'] == 'Depthwise':
          params['max_depth'] = trial.suggest_int('max_depth', 2, 12)
        
        model = cat.CatBoostClassifier(
            **params,
            cat_features=self.cat_features,
            verbose=False,
            random_state=2025,
            eval_metric = 'AUC',
            task_type = 'CPU',
            thread_count=-1  # Используем все ядра
        )
        model.fit(
               X_train, y_train,
               eval_set=(X_val, y_val),
               verbose=0
          )
     #    print("="*20, self.n_segments)
        if self.n_segments == 2:
          # print("="*20, self.n_segments, "in 2")
          y_pred = model.predict_proba(X_val)[:, 1]
          auc = metrics.roc_auc_score(y_val, y_pred)
        else:
          auc = metrics.roc_auc_score(y_val, model.predict_proba(X_val), multi_class='ovr')
        
        return auc
    
    def fit(self, X, y, n_trials=50):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: self._objective(t, X, y), n_trials=n_trials)
        
        self.best_params = study.best_params
        self.model = cat.CatBoostClassifier(
            **self.best_params,
            cat_features=self.cat_features,
            random_state=2025,
            eval_metric = 'AUC',
            task_type = 'CPU',
            thread_count=-1,
            verbose=False
        )
        self.model.fit(X, y)
        
        # Оценка качества
        pred = self.model.predict(X)
        print("Classification Report: TRAIN")
        print(metrics.classification_report(y, pred))
        # если бинарный сегмент то так
        if self.n_segments == 2:
          print("ROC AUC =", metrics.roc_auc_score(y, self.model.predict_proba(X)[:, 1]))
        # если сегментов больше 2
        else:
           print("ROC AUC =", metrics.roc_auc_score(y, self.model.predict_proba(X), multi_class='ovr'))

        return self
    
    def predict_proba(self, X):
        """Возвращает вероятности классов"""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Возвращает предсказанные классы"""
        return self.model.predict(X)