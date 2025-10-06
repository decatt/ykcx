from dataloader import ExcelDataLoader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import json
from datetime import datetime

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 加载和预处理数据
loader = ExcelDataLoader(file_path="data.xlsx")
if loader.load_data() and loader.preprocess_data():
    x, y = loader.get_train_data()
    print(f"数据加载成功: 特征维度 {x.shape}, 标签维度 {y.shape}")
else:
    print("数据加载失败！")
    exit()

# 2. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print(f"\n训练集大小: {x_train.shape[0]}, 测试集大小: {x_test.shape[0]}")

# 3. 创建DMatrix（用于原生XGBoost）
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

# 4. 定义模型参数
params = {
    "max_depth": 5,
    "eta": 0.1,
    "objective": "reg:squarederror",
    "random_state": 42
}

# 5. 训练XGBoost回归模型
print("\n开始训练模型...")
reg_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)

reg_model.fit(
    x_train, y_train,
    eval_set=[(x_train, y_train), (x_test, y_test)],
    verbose=False
)
print("模型训练完成！")

# 6. 进行预测
y_train_pred = reg_model.predict(x_train)
y_test_pred = reg_model.predict(x_test)

# 7. 计算评估指标
def evaluate_model(y_true, y_pred, dataset_name=""):
    """计算并打印模型评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    print(f"\n{'='*50}")
    print(f"{dataset_name}评估结果:")
    print(f"{'='*50}")
    print(f"均方误差 (MSE):        {mse:.6f}")
    print(f"均方根误差 (RMSE):     {rmse:.6f}")
    print(f"平均绝对误差 (MAE):    {mae:.6f}")
    print(f"R² 决定系数:           {r2:.6f}")
    print(f"平均绝对百分比误差:    {mape:.2f}%")
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }

train_metrics = evaluate_model(y_train, y_train_pred, "训练集")
test_metrics = evaluate_model(y_test, y_test_pred, "测试集")

# 8. 保存模型
model_filename = f"xgboost_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
reg_model.save_model(model_filename.replace('.pkl', '.json'))
joblib.dump(reg_model, model_filename)
print(f"\n模型已保存为: {model_filename}")
print(f"原生格式保存为: {model_filename.replace('.pkl', '.json')}")

# 9. 保存评估结果
results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_params': params,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'feature_importance': {
        f'feature_{i}': float(imp) 
        for i, imp in enumerate(reg_model.feature_importances_)
    }
}

with open('model_evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print("评估结果已保存为: model_evaluation_results.json")

# 10. 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 10.1 训练集预测 vs 实际
axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5, s=20)
axes[0, 0].plot([y_train.min(), y_train.max()], 
                [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual value', fontsize=12)
axes[0, 0].set_ylabel('Predicted value', fontsize=12)
axes[0, 0].set_title(f'Training set prediction results (R^2={train_metrics["r2"]:.4f})', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)

# 10.2 测试集预测 vs 实际
axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual value', fontsize=12)
axes[0, 1].set_ylabel('Predicted value', fontsize=12)
axes[0, 1].set_title(f'Test set prediction results (R^2={test_metrics["r2"]:.4f})', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# 10.3 残差分布
residuals_test = y_test - y_test_pred
axes[1, 0].hist(residuals_test, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('residual', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Test set residual distribution', fontsize=14)
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].grid(True, alpha=0.3)

# 10.4 特征重要性
feature_importance = reg_model.feature_importances_
features = [f'features{i+1}' for i in range(len(feature_importance))]
sorted_idx = np.argsort(feature_importance)[-10:]  # 显示前10个最重要的特征

axes[1, 1].barh(range(len(sorted_idx)), feature_importance[sorted_idx])
axes[1, 1].set_yticks(range(len(sorted_idx)))
axes[1, 1].set_yticklabels([features[i] for i in sorted_idx])
axes[1, 1].set_xlabel('Importance score', fontsize=12)
axes[1, 1].set_title('Importance of the top 10 features', fontsize=14)
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plot_filename = f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"可视化结果已保存为: {plot_filename}")
plt.show()
