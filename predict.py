import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
import pickle
import os

MODELS_DIR = 'open_source_model/models'
STAGE1_NN_MODEL_PATH = os.path.join(MODELS_DIR, 'stage1_nn_model.h5')
STAGE1_SCALER_PATH = os.path.join(MODELS_DIR, 'stage1_scaler.pkl')
STAGE2_RF_MODEL_PATH = os.path.join(MODELS_DIR, 'stage2_rf_model.pkl')
STAGE3_XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'stage3_xgb_model.pkl')

FEATURE_LIST = [
    "AMH", "扳机日E2", "雄烯二酮（0.3-3.3ng0ml）", "AFC", "SHBG（nmol0l）",
    "扳机日14mm卵泡数", "75g三小时血糖", "75g二小时血糖", "75g半小时血糖", "BMI",
    "FAI", "HDL-C", "LH", "不孕年限", "同型半胱氨酸（5-15umol0l）", "年龄",
    "扳机日P", "扳机日内膜厚度", "睾酮(0.14-0.76ng0ml)", "空腹胰岛素", "餐后3小时胰岛素"
]

try:
    stage1_nn_model = tf.keras.models.load_model(STAGE1_NN_MODEL_PATH)
    with open(STAGE1_SCALER_PATH, 'rb') as f:
        stage1_scaler = pickle.load(f)
    with open(STAGE2_RF_MODEL_PATH, 'rb') as f:
        stage2_rf_model = pickle.load(f)
    with open(STAGE3_XGB_MODEL_PATH, 'rb') as f:
        stage3_xgb_model = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading models: {e}.")
    print("Please ensure the 'models' directory and all model files are present.")
    stage1_nn_model = stage1_scaler = stage2_rf_model = stage3_xgb_model = None

def make_predictions(input_df: pd.DataFrame):
    if stage1_nn_model is None:
        raise RuntimeError("Models are not loaded. Cannot make predictions.")
        
    if not all(feature in input_df.columns for feature in FEATURE_LIST):
        missing = [f for f in FEATURE_LIST if f not in input_df.columns]
        raise ValueError(f"Input data is missing the following required features: {missing}")

    X_new = input_df[FEATURE_LIST]

    X_new_scaled = stage1_scaler.transform(X_new)
    prob_stage1 = stage1_nn_model.predict(X_new_scaled).flatten()

    X_new_aug = X_new.copy()
    X_new_aug['stage1_nn_prob'] = prob_stage1
    
    prob_stage2 = stage2_rf_model.predict_proba(X_new_aug)[:, 1]
    
    prob_stage3 = stage3_xgb_model.predict_proba(X_new_aug)[:, 1]

    results_df = pd.DataFrame({
        'Prediction_A': prob_stage1,
        'Prediction_B': prob_stage2,
        'Prediction_C': prob_stage3
    })
    
    return results_df

if __name__ == '__main__':
    sample_data = pd.DataFrame([{
        "AMH": 2.5, "扳机日E2": 2000, "雄烯二酮（0.3-3.3ng0ml）": 1.0, 
        "AFC": 15, "SHBG（nmol0l）": 30, "扳机日14mm卵泡数": 5, 
        "75g三小时血糖": 6.0, "75g二小时血糖": 7.0, "75g半小时血糖": 8.0, 
        "BMI": 22, "FAI": 3.0, "HDL-C": 1.5, "LH": 10, 
        "不孕年限": 2, "同型半胱氨酸（5-15umol0l）": 10, "年龄": 30, 
        "扳机日P": 1.0, "扳机日内膜厚度": 10, "睾酮(0.14-0.76ng0ml)": 0.5, 
        "空腹胰岛素": 8, "餐后3小时胰岛素": 50
    }])

    print("\n--- Making prediction on sample data ---")
    
    try:
        predictions = make_predictions(sample_data)
        if predictions is not None:
            print("\nPredicted Probabilities:")
            print(predictions)
    except (ValueError, RuntimeError) as e:
        print(f"An error occurred: {e}") 