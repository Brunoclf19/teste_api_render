# rossmann/Rossmann.py
import os
import pickle
import pandas as pd
import numpy as np
import inflection
import datetime
import math

class Rossmann(object):
    def __init__(self):
        # raiz do projeto
        self.home_path = r''

        # use os.path.join para evitar caminho inválido
        param_dir = os.path.join(self.home_path, 'parameter')
        self.competition_distance_scaler    = pickle.load(open(os.path.join(param_dir, 'competition_distance_scaler.pkl'), 'rb'))
        self.competition_time_month_scaler  = pickle.load(open(os.path.join(param_dir, 'competition_time_month_scaler.pkl'), 'rb'))
        self.promo_time_week_scaler         = pickle.load(open(os.path.join(param_dir, 'promo_time_week_scaler.pkl'), 'rb'))
        self.year_scaler                    = pickle.load(open(os.path.join(param_dir, 'year_scaler.pkl'), 'rb'))
        self.store_type_scaler              = pickle.load(open(os.path.join(param_dir, 'store_type_scaler.pkl'), 'rb'))

    def data_cleaning(self, df1):
        # 1) renomear SOMENTE as colunas necessárias (sem mexer na ordem do DF)
        rename_map = {
            'Store': 'store',
            'DayOfWeek': 'day_of_week',
            'Date': 'date',
            'Open': 'open',
            'Promo': 'promo',
            'StateHoliday': 'state_holiday',
            'SchoolHoliday': 'school_holiday',
            'StoreType': 'store_type',
            'Assortment': 'assortment',
            'CompetitionDistance': 'competition_distance',
            'CompetitionOpenSinceMonth': 'competition_open_since_month',
            'CompetitionOpenSinceYear': 'competition_open_since_year',
            'Promo2': 'promo2',
            'Promo2SinceWeek': 'promo2_since_week',
            'Promo2SinceYear': 'promo2_since_year',
            'PromoInterval': 'promo_interval',
        }
        df1 = df1.rename(columns=rename_map)

        # 2) garantir a coluna 'date' e converter de forma segura
        if 'date' not in df1.columns and 'Date' in df1.columns:
            df1 = df1.rename(columns={'Date': 'date'})

        df1['date'] = pd.to_datetime(df1['date'], errors='coerce')

        # validação rápida para evitar .dt em objeto
        if df1['date'].isna().all():
            raise ValueError("Nenhuma data válida após pd.to_datetime; verifique a coluna 'Date' do payload.")

        # ===== resto do seu código de NA/encoding, ajustado com cuidados =====
        # competition_distance
        df1['competition_distance'] = df1['competition_distance'].apply(
            lambda x: 200000.0 if (pd.isna(x)) else x
        )

        # competition_open_since_month/year
        df1['competition_open_since_month'] = df1.apply(
            lambda x: x['date'].month if pd.isna(x['competition_open_since_month']) else x['competition_open_since_month'],
            axis=1
        )
        df1['competition_open_since_year'] = df1.apply(
            lambda x: x['date'].year if pd.isna(x['competition_open_since_year']) else x['competition_open_since_year'],
            axis=1
        )

        # promo2_since_week/year
        df1['promo2_since_week'] = df1.apply(
            lambda x: x['date'].isocalendar().week if pd.isna(x['promo2_since_week']) else x['promo2_since_week'],
            axis=1
        )
        df1['promo2_since_year'] = df1.apply(
            lambda x: x['date'].year if pd.isna(x['promo2_since_year']) else x['promo2_since_year'],
            axis=1
        )

        # promo_interval (use inglês porque o dataset original é em EN)
        month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        df1['promo_interval'] = df1['promo_interval'].fillna('')
        df1['month_map'] = df1['date'].dt.month.map(month_map)
        df1['is_promo'] = df1.apply(
            lambda x: 1 if (x['promo_interval'] and str(x['month_map']) in str(x['promo_interval']).split(',')) else 0,
            axis=1
        )

        # types
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year']  = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

        return df1


    def feature_engineering(self, df2):
        # defensivo: garantir datetime
        if not pd.api.types.is_datetime64_any_dtype(df2['date']):
            df2['date'] = pd.to_datetime(df2['date'], errors='coerce')

        df2['year']  = df2['date'].dt.year
        df2['month'] = df2['date'].dt.month
        df2['day']   = df2['date'].dt.day
        df2['week_of_year'] = df2['date'].dt.isocalendar().week
        df2['year_week']    = df2['date'].dt.strftime('%Y-%W')

        # competition since
        df2['competition_since'] = df2.apply(
            lambda x: datetime.datetime(
                year=int(x['competition_open_since_year']),
                month=int(x['competition_open_since_month']),
                day=1
            ),
            axis=1
        )

        # meses de competição (diferença em dias // 30)
        comp_days = (df2['date'] - df2['competition_since']).dt.days
        comp_days = comp_days.fillna(0).astype(int)
        df2['competition_time_month'] = comp_days // 30


        # promo since
        df2['promo_since'] = df2.apply(
            lambda x: datetime.datetime(
                year=int(x['promo2_since_year']),
                month=1,
                day=1
            ) + datetime.timedelta(weeks=int(x['promo2_since_week']) - 1),
            axis=1
        )

        # semanas de promoção (diferença em dias // 7)
        promo_days = (df2['date'] - df2['promo_since']).dt.days
        promo_days = promo_days.fillna(0).astype(int)
        df2['promo_time_week'] = promo_days // 7

        df2['assortment'] = df2['assortment'].map({'a':'basic','b':'extra','c':'extended'})
        df2['state_holiday'] = df2['state_holiday'].map({'a':'public_holiday','b':'easter_holiday','c':'christmas'}).fillna('regular_day')

        df2 = df2[df2['open'] != 0]
        df2 = df2.drop(['open','promo_interval','month_map'], axis=1, errors='ignore')

        return df2


    def data_preparation(self, df5):
        # escalonamento
        df5['competition_distance']     = self.competition_distance_scaler.transform(df5[['competition_distance']].values)
        df5['competition_time_month']   = self.competition_time_month_scaler.transform(df5[['competition_time_month']].values)
        df5['promo_time_week']          = self.promo_time_week_scaler.transform(df5[['promo_time_week']].values)
        df5['year']                     = self.year_scaler.transform(df5[['year']].values)

        # encoding
        df5 = pd.get_dummies(df5, columns=['state_holiday'], prefix='state_holiday')
        df5['store_type'] = self.store_type_scaler.transform(df5['store_type'])
        df5['assortment'] = df5['assortment'].map({'basic':1,'extra':2,'extended':3})

        # ciclicidade
        df5['day_of_week_sin'] = np.sin(df5['day_of_week'] * (2 * np.pi / 7))
        df5['day_of_week_cos'] = np.cos(df5['day_of_week'] * (2 * np.pi / 7))
        df5['month_sin'] = np.sin(df5['month'] * (2 * np.pi / 12))
        df5['month_cos'] = np.cos(df5['month'] * (2 * np.pi / 12))
        df5['day_sin'] = np.sin(df5['day'] * (2 * np.pi / 30))
        df5['day_cos'] = np.cos(df5['day'] * (2 * np.pi / 30))
        df5['week_of_year_sin'] = np.sin(df5['week_of_year'] * (2 * np.pi / 52))
        df5['week_of_year_cos'] = np.cos(df5['week_of_year'] * (2 * np.pi / 52))

        # dentro de Rossmann.data_preparation
        cols_selected = [
            'store','promo','store_type','assortment','competition_distance',
            'competition_open_since_month','competition_open_since_year','promo2',
            'promo2_since_week','promo2_since_year','competition_time_month','promo_time_week',
            'day_of_week_sin','day_of_week_cos',
            'week_of_year_cos','week_of_year_sin',   # <- ordem esperada pelo modelo
            'month_cos','month_sin',
            'day_sin','day_cos'
        ]
        return df5[cols_selected]


    def get_prediction(self, model, original_data, test_data):
        # Garante que as colunas estejam na MESMA ordem do treino
        expected = model.get_booster().feature_names

        missing = [c for c in expected if c not in test_data.columns]
        extra   = [c for c in test_data.columns if c not in expected]

        if missing or extra:
            raise ValueError(f"Colunas incompatíveis. Faltando: {missing} | Sobrando: {extra}")

        test_data = test_data[expected]  # reordena exatamente

        pred = model.predict(test_data)
        original_data = original_data.copy()
        original_data['predictions'] = np.expm1(pred)
        return original_data.to_json(orient='records', date_format='iso')

