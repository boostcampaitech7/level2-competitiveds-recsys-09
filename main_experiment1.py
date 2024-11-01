import argparse
import yaml
import pandas as pd
from src.data.data_loader import download_data, extract_data, load_raw_data, load_preprocessed_data
from src.features.feature_engineering_1 import feature_engineering
from src.models.train_model_eval import train_and_evaluate_models
from src.models.train_model_batch import train_and_predict_models
from src.utils.submission import submission_to_csv

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description='Run project with optional MAE evaluation')
    parser.add_argument('--eval-mae', action='store_true', help='Evaluate MAE on holdout set')
    args = parser.parse_args()
    
    config = load_config()

    # 데이터 로드
    train_data, test_data , submission, _, _, _, _ = load_preprocessed_data()

    # 모델별 파라미터 설정
    params_lgb = config['model_params']['lightgbm']
    params_cat = config['model_params']['catboost']
    params_xgb = config['model_params']['xgboost']
   
    try:
        
        # duplicated data 제거
        dup_df = train_data[train_data.drop(columns="index").duplicated()]
        train_data = train_data.drop(index=dup_df.index)

        # Feature Engineering 실행
        train_processed, test_processed = feature_engineering(train_data, test_data)
        
        # 검증용 MAE 평가 활성화 여부 확인 (holdout MAE가 필요할 때만 실행)
        if args.eval_mae:
            print("Holdout MAE 평가를 위해 holdout 데이터셋을 분리합니다.")
            
            # holdout 데이터 분리
            holdout_start, holdout_end = 202307, 202312
            holdout_data = train_processed[(train_processed['contract_year_month'] >= holdout_start) & (train_processed['contract_year_month'] <= holdout_end)]
            train_data = train_processed[~((train_processed['contract_year_month'] >= holdout_start) & (train_processed['contract_year_month'] <= holdout_end))]

            # 학습과 검증용 데이터셋 설정
            X_train = train_data.drop(columns=['deposit', 'deposit_per_area'])
            y_train = train_data['deposit_per_area']
            X_holdout = holdout_data.drop(columns=['deposit', 'deposit_per_area'])
            y_holdout = holdout_data['deposit_per_area']
            
            X_train.drop('contract_year_month', axis=1, inplace=True)
            X_holdout.drop('contract_year_month', axis=1, inplace=True)

            # 모델 학습 및 MAE 평가
            mae_lgb, mae_xgb, mae_cat = train_and_evaluate_models(
                X_train, y_train, X_holdout, y_holdout, params_lgb, params_xgb, params_cat
            )
            print("Holdout 데이터셋 성능:")
            print(f"LightGBM MAE: {mae_lgb:.2f}")
            print(f"XGBoost MAE: {mae_xgb:.2f}")
            print(f"CatBoost MAE: {mae_cat:.2f}")

        # 전체 데이터 학습 및 예측 
        X_train = train_processed.drop(columns=['deposit', 'deposit_per_area'])
        y_train = train_processed['deposit_per_area']
        X_test = test_processed.copy()
        
        X_train.drop('contract_year_month', axis=1, inplace=True)
        X_test.drop('contract_year_month', axis=1, inplace=True)
        
        lgb_test_pred_origin, xgb_test_pred_origin, cat_test_pred_origin = train_and_predict_models(
            X_train, y_train, X_test, params_lgb, params_xgb, params_cat
        )
        
        ensemble_pred = xgb_test_pred_origin * 0.66 + lgb_test_pred_origin * 0.23 + cat_test_pred_origin * 0.11
        # submission_to_csv(sample_submission, ensemble_pred)
        print(ensemble_pred.describe())
    except Exception as e:
        print(f"An error occurred during feature engineering or model training: {e}")

if __name__ == '__main__':
    main()
