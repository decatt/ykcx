import joblib
import numpy as np

model = joblib.load("xgboost_model_20251005_230215.pkl")
X_new = np.array([1,2,0,3.33,3.27,0,1,1,1.96,2,25.00,1,4.08,21.46,43.61,170,45.58,80,3.44,11.9,-0.26])

def compute_t(x=X_new):
    # x 形状 (20,)
    # 复制10次
    X = np.tile(x, (10, 1))
    # 将第10列改为[25,26,27,...,34]
    X[:,10] = np.arange(25,35)
    # 使用模型预测
    y_pred = model.predict(X)
    # 返回形状 (10, 2) 的结果[25,26,27,...,34],y_pred
    return np.column_stack((X[:,10], y_pred))