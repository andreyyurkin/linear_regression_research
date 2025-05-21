import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder

class DataPreprocessor:
    def __init__(
        self,
        target_col: str,
        num_features: list = None,
        cat_features: list = None,
        cat_encoder_type: str = "target",  # "target" или "ohe"
        fillna_num_strategy: str = "zero",  # "zero" или "mean"
        scaler_type: str = "minmax",  # "minmax" или "standard"
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Инициализация препроцессора.
        
        :param target_col: Название целевой переменной.
        :param num_features: Список числовых признаков (если None - автоопределение).
        :param cat_features: Список категориальных признаков (если None - автоопределение).
        :param cat_encoder_type: Тип кодирования категориальных признаков ("target" или "ohe").
        :param fillna_num_strategy: Стратегия заполнения пропусков в числовых признаках ("zero" или "mean").
        :param scaler_type: Тип масштабирования числовых признаков ("minmax" или "standard").
        :param test_size: Размер тестовой выборки (0.2 = 20%).
        :param random_state: Random seed для воспроизводимости.
        """
        self.target_col = target_col
        self.num_features = num_features
        self.cat_features = cat_features
        self.cat_encoder_type = cat_encoder_type
        self.fillna_num_strategy = fillna_num_strategy
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.random_state = random_state
        
        # Инициализация объектов для трансформации
        self.scaler = None
        self.cat_encoder = None
        
    def _detect_features(self, df: pd.DataFrame) -> None:
        """Автоматическое определение числовых и категориальных признаков."""
        if self.num_features is None:
            self.num_features = list(df.select_dtypes(include=[np.number]).columns)
            self.num_features.remove(self.target_col)  # Исключаем target из числовых
            
        if self.cat_features is None:
            self.cat_features = list(df.select_dtypes(include=["object", "category"]).columns)
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
     """Заполнение пропущенных значений."""
     df_filled = df.copy()
     
     # Заполнение числовых признаков
     if self.num_features:
          if self.fillna_num_strategy == "zero":
               df_filled[self.num_features] = df_filled[self.num_features].fillna(0)
          elif self.fillna_num_strategy == "mean":
               df_filled[self.num_features] = df_filled[self.num_features].fillna(
                    df_filled[self.num_features].mean()
               )
     
     # Заполнение категориальных признаков
     if self.cat_features:
          for col in self.cat_features:
               # Преобразуем строковый тип, если это Categorical
               if pd.api.types.is_categorical_dtype(df_filled[col]):
                    df_filled[col] = df_filled[col].astype(str)
               df_filled[col] = df_filled[col].fillna("Other")
     
     return df_filled
    
    def _encode_categorical(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> tuple:
          """Кодирование категориальных признаков."""
          if not self.cat_features:
               return X_train, X_test

          if self.cat_encoder_type == "target":
               self.cat_encoder = TargetEncoder(cols=self.cat_features)
               self.cat_encoder.fit(X_train, y_train)
               X_train_encoded = self.cat_encoder.transform(X_train)
               X_test_encoded = self.cat_encoder.transform(X_test)
               
          elif self.cat_encoder_type == "ohe":
               self.cat_encoder = OneHotEncoder(drop='first', sparse_output=False)
               
               # Кодируем только категориальные признаки
               X_train_encoded = X_train.copy()
               X_test_encoded = X_test.copy()
               
               ohe_result_train = self.cat_encoder.fit_transform(X_train[self.cat_features])
               ohe_result_test = self.cat_encoder.transform(X_test[self.cat_features])
               
               # Создаем названия для новых колонок
               feature_names = self.cat_encoder.get_feature_names_out(self.cat_features)
               
               # Удаляем старые категориальные признаки
               X_train_encoded = X_train_encoded.drop(columns=self.cat_features)
               X_test_encoded = X_test_encoded.drop(columns=self.cat_features)
               
               # Добавляем закодированные признаки
               X_train_encoded[feature_names] = ohe_result_train
               X_test_encoded[feature_names] = ohe_result_test
               
          return X_train_encoded, X_test_encoded
    
    def _scale_numerical(self, X_train: pd.DataFrame, 
                         X_test: pd.DataFrame) -> tuple:
        """Масштабирование числовых признаков."""
        if not self.num_features:
            return X_train, X_test  # Если нет числовых признаков
        
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
            
        if self.cat_encoder_type == 'ohe':
             X_train[self.num_features] = self.scaler.fit_transform(X_train[self.num_features])
             X_test[self.num_features] = self.scaler.transform(X_test[self.num_features])
        elif self.cat_encoder_type == 'target':
             X_train[self.num_features+self.cat_features] = self.scaler\
                         .fit_transform(X_train[self.num_features+self.cat_features])
             X_test[self.num_features+self.cat_features] = self.scaler\
                         .transform(X_test[self.num_features+self.cat_features])
        else:
            raise ValueError("Wrong scaler type!")
        
        return X_train, X_test
    
    def fit_transform(self, df: pd.DataFrame) -> tuple:
     """
     Основной метод: разделяет данные, заполняет пропуски, кодирует категории, масштабирует числовые признаки.
     
     :return: (train_processed, test_processed) - обработанные DataFrame с сохраненным target
     """
     # Определяем признаки, если не заданы вручную
     self._detect_features(df)
     
     # Заполняем пропуски
     df_filled = self._fill_missing_values(df)
     
     # Разделяем на train/test и сразу сбрасываем индексы
     X = df_filled.drop(columns=[self.target_col])
     y = df_filled[self.target_col]
     
     X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=self.test_size, random_state=self.random_state
     )
     X_train = X_train.reset_index(drop=True)
     X_test = X_test.reset_index(drop=True)
     y_train = y_train.reset_index(drop=True)
     y_test = y_test.reset_index(drop=True)
     
     # Кодируем категориальные признаки (теперь индексы уже согласованы)
     saved_cat_features = {i: f'CAT_{i}' for i in self.cat_features}
     X_train_cat, X_test_cat = X_train[self.cat_features], X_test[self.cat_features]
     X_train_cat = X_train_cat.rename(columns = saved_cat_features)
     X_test_cat = X_test_cat.rename(columns = saved_cat_features)

     X_train_encoded, X_test_encoded = self._encode_categorical(X_train, X_test, y_train)
     
     # Масштабируем числовые признаки
     X_train_scaled, X_test_scaled = self._scale_numerical(X_train_encoded, X_test_encoded)

     # Собираем финальные датафреймы
     train_processed = pd.concat([X_train_scaled, X_train_cat, y_train], axis=1)
     test_processed = pd.concat([X_test_scaled, X_test_cat, y_test], axis=1)
     
     return train_processed, test_processed