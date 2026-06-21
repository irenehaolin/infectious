# === 1. 导入库 ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# === 2. 读取并处理数据 ===
file_path = "data1_lagged_final_vars.xlsx"
if not os.path.exists(file_path):
    file_path = os.path.join(os.path.dirname(sys.argv[0]), file_path)
if not os.path.exists(file_path):
    raise FileNotFoundError(f"数据文件 {file_path} 未找到，请确保文件存在于当前目录或脚本目录")
data = pd.read_excel(file_path)[['Date', 'Influenza']]
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# log变换 + 归一化 ILI%
data['log_Influenza'] = np.log1p(data['Influenza'])

# === 3. 拆分训练和测试 ===
train_data = data.iloc[:-26].copy()
test_data = data.iloc[-26:].copy()

# === 4. ===
# 初始化归一化器
ili_scaler = MinMaxScaler()

# 训练集归一化
train_data['log_Influenza_scaled'] = ili_scaler.fit_transform(train_data[['log_Influenza']])

# 测试集使用训练集的归一化器进行转换
test_data['log_Influenza_scaled'] = ili_scaler.transform(test_data[['log_Influenza']])

# ================================================================
# === 5. SARIMAX参数网格搜索 ===
# ================================================================
def sarimax_grid_search(data, p_range, d_range, q_range, P_range, D_range, Q_range, s):
    """
    SARIMAX参数网格搜索，使用AIC准则选择最优参数
    
    参数:
        data: 时间序列数据
        p_range: AR阶数范围 (e.g., [0,1,2])
        d_range: 差分阶数范围 (e.g., [0,2])
        q_range: MA阶数范围 (e.g., [0,1,2])
        P_range: 季节性AR阶数范围 (e.g., [0,1,2])
        D_range: 季节性差分阶数范围 (e.g., [0,1])
        Q_range: 季节性MA阶数范围 (e.g., [0,1,2])
        s: 季节周期 (e.g., 52 for weekly data)
    
    返回:
        best_order: 最优非季节性参数 (p,d,q)
        best_seasonal_order: 最优季节性参数 (P,D,Q,s)
        best_model: 最优模型
        results_df: 所有组合的结果DataFrame
    """
    print("\n" + "="*60)
    print("SARIMAX参数网格搜索")
    print("="*60)
    
    # 生成所有参数组合
    orders = list(product(p_range, d_range, q_range))
    seasonal_orders = list(product(P_range, D_range, Q_range, [s]))
    
    results = []
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None
    
    total_combinations = len(orders) * len(seasonal_orders)
    current = 0
    
    print(f"总共需要测试 {total_combinations} 种参数组合...")
    print("-" * 60)
    
    for order in orders:
        for seasonal_order in seasonal_orders:
            current += 1
            try:
                # 构建SARIMAX模型
                model = SARIMAX(data, 
                               order=order, 
                               seasonal_order=seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                
                # 拟合模型
                results_model = model.fit(disp=False, maxiter=200)
                
                # 记录结果
                results.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'AIC': results_model.aic,
                    'BIC': results_model.bic,
                    'HQIC': results_model.hqic,
                    'params': results_model.params.shape[0]
                })
                
                # 更新最优模型
                if results_model.aic < best_aic:
                    best_aic = results_model.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_model = results_model
                    
                    print(f"[{current}/{total_combinations}] 新的最优参数: order={order}, seasonal_order={seasonal_order}, AIC={results_model.aic:.2f}")
                
            except Exception as e:
                # 模型拟合失败，跳过该组合
                pass
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AIC')
    
    print("\n" + "="*60)
    print("网格搜索完成！")
    print(f"最优参数: order={best_order}, seasonal_order={best_seasonal_order}")
    print(f"最优AIC值: {best_aic:.2f}")
    print("="*60)
    
    # 保存搜索结果
    results_df.to_csv('sarimax_grid_search_results.csv', index=False)
    print("\n搜索结果已保存到: sarimax_grid_search_results.csv")
    
    return best_order, best_seasonal_order, best_model, results_df

# 定义参数搜索范围
print("\n正在设置参数搜索范围...")
p_range = [0, 1, 2]  # AR阶数
d_range = [0, 1]      # 差分阶数
q_range = [0, 1, 2]  # MA阶数
P_range = [0, 1, 2]  # 季节性AR阶数
D_range = [0, 1]     # 季节性差分阶数
Q_range = [0, 1, 2]  # 季节性MA阶数
seasonal_period = 52  # 周数据，季节周期为52周

print(f"参数搜索范围:")
print(f"  p (AR阶数): {p_range}")
print(f"  d (差分阶数): {d_range}")
print(f"  q (MA阶数): {q_range}")
print(f"  P (季节性AR阶数): {P_range}")
print(f"  D (季节性差分阶数): {D_range}")
print(f"  Q (季节性MA阶数): {Q_range}")
print(f"  季节周期 s: {seasonal_period}")

# 执行网格搜索
best_order, best_seasonal_order, sarimax_fit, search_results = sarimax_grid_search(
    train_data['log_Influenza_scaled'],
    p_range, d_range, q_range,
    P_range, D_range, Q_range,
    seasonal_period
)

# 显示Top 10参数组合
print("\n" + "="*60)
print("Top 10 参数组合 (按AIC排序)")
print("="*60)
print(search_results.head(10).to_string(index=False))

# 绘制网格搜索结果
plt.figure(figsize=(12, 6))
top_20 = search_results.head(20)
plt.barh(range(len(top_20)), top_20['AIC'].values, color='steelblue')
plt.yticks(range(len(top_20)), 
          [f"({o[0]},{o[1]},{o[2]})x({s[0]},{s[1]},{s[2]},{s[3]})" 
           for o, s in zip(top_20['order'], top_20['seasonal_order'])])
plt.xlabel('AIC Value')
plt.ylabel('Parameter Combination (order)x(seasonal_order)')
plt.title('SARIMAX Grid Search Results - Top 20 Models by AIC')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('sarimax_grid_search_results.pdf', format='pdf', dpi=300)
plt.close()
print("\n网格搜索可视化已保存到: sarimax_grid_search_results.pdf")

# 使用最优参数重新训练模型（如果需要）
print("\n" + "="*60)
print("使用最优参数训练最终模型")
print("="*60)
print(f"最优参数: order={best_order}, seasonal_order={best_seasonal_order}")

# 重新构建模型
sarimax_model = SARIMAX(train_data['log_Influenza_scaled'],
                       order=best_order,
                       seasonal_order=best_seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
sarimax_fit = sarimax_model.fit(disp=False)
print(sarimax_fit.summary())

# === 6. 训练 + 测试预测 ===
sarimax_train_pred = sarimax_fit.predict(start=0, end=len(train_data)-1)

# 使用 get_forecast 获取测试集预测值和预测区间（95%）
sarimax_forecast = sarimax_fit.get_forecast(steps=len(test_data))
sarimax_test_pred = sarimax_forecast.predicted_mean
test_conf_int = sarimax_forecast.conf_int(alpha=0.05)  # 95% 预测区间

# === 8. 反归一化 + 反对数 ===
# 训练集
train_pred_log = ili_scaler.inverse_transform(sarimax_train_pred.values.reshape(-1, 1))
train_pred = np.expm1(train_pred_log)

# 测试集
test_pred_log = ili_scaler.inverse_transform(sarimax_test_pred.values.reshape(-1, 1))
test_pred = np.expm1(test_pred_log)

# 反归一化预测区间
test_conf_int_log_lower = ili_scaler.inverse_transform(test_conf_int.iloc[:, 0].values.reshape(-1, 1))
test_conf_int_log_upper = ili_scaler.inverse_transform(test_conf_int.iloc[:, 1].values.reshape(-1, 1))
test_conf_int_lower = np.expm1(test_conf_int_log_lower)
test_conf_int_upper = np.expm1(test_conf_int_log_upper)

# 实际值
train_actual = np.expm1(ili_scaler.inverse_transform(train_data['log_Influenza_scaled'].values.reshape(-1, 1)))
test_actual = np.expm1(ili_scaler.inverse_transform(test_data['log_Influenza_scaled'].values.reshape(-1, 1)))

# === 9. 模型评估指标 ===
# 训练集指标
mae_train = mean_absolute_error(train_actual, train_pred)
rmse_train = np.sqrt(mean_squared_error(train_actual, train_pred))
r2_train = r2_score(train_actual, train_pred)

# 测试集指标
mae_test = mean_absolute_error(test_actual, test_pred)
rmse_test = np.sqrt(mean_squared_error(test_actual, test_pred))
r2_test = r2_score(test_actual, test_pred)

print("\n【SARIMAX 模型评估指标】")
print(f"训练集: MAE = {mae_train:.4f}, RMSE = {rmse_train:.4f}, R² = {r2_train:.4f}")
print(f"测试集: MAE = {mae_test:.4f}, RMSE = {rmse_test:.4f}, R² = {r2_test:.4f}")

# === 10. 绘制完整趋势图 ===
full_pred = np.concatenate([train_pred.flatten(), test_pred.flatten()])
full_actual = data['Influenza'].values
date_range = data.index

plt.figure(figsize=(14, 6), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

plt.plot(date_range, full_actual, label='Actual')
plt.plot(date_range, full_pred, label='Predicted')
plt.axvline(test_data.index[0], color='red', linestyle='--', label='Test point')
plt.title(f'SARIMAX{best_order}x{best_seasonal_order[:3]} Model for Trend Forecasting')
plt.xlabel('Time')
plt.ylabel('Influenza')
plt.legend()
plt.grid(True, alpha=0.5, color='gray')
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('SARIMAX_trend.pdf', format='pdf', dpi=300)
plt.close()

# ======训练集拟合图=====
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_actual, label='Training Actual')
plt.plot(train_data.index, train_pred.flatten(), label='Training Fitted')
plt.title(f'Training Set Fitting SARIMAX{best_order}x{best_seasonal_order[:3]} Model')
plt.xlabel('Time')
plt.ylabel('Influenza incidence')
plt.legend()
plt.grid()
plt.savefig('SARIMAX_train.pdf', format='pdf', dpi=300)
plt.close()

# === 11. 测试集预测图 + 预测区间 ===
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_actual.flatten(), label='Actual')
plt.plot(test_data.index, test_pred.flatten(), label='Predicted', color='orange')
plt.fill_between(test_data.index,
                 test_conf_int_lower.flatten(),
                 test_conf_int_upper.flatten(),
                 color='gray', alpha=0.3,
                 label='95% Prediction Interval')
plt.title(f'Test Set Prediction SARIMAX{best_order}x{best_seasonal_order[:3]} with 95% CI')
plt.xlabel('Time')
plt.ylabel('Influenza incidence')
plt.legend()
plt.grid()
plt.savefig('SARIMAX_test.pdf', format='pdf', dpi=300)
plt.close()

print("所有图表已保存完成！")

# === 12. 保存最优参数 ===
with open('sarimax_best_params.txt', 'w') as f:
    f.write("SARIMAX最优参数\n")
    f.write("="*50 + "\n")
    f.write(f"非季节性参数 order (p,d,q): {best_order}\n")
    f.write(f"季节性参数 seasonal_order (P,D,Q,s): {best_seasonal_order}\n")
    f.write(f"最优AIC值: {sarimax_fit.aic:.2f}\n")
    f.write(f"最优BIC值: {sarimax_fit.bic:.2f}\n")
    f.write("\n模型评估指标\n")
    f.write("="*50 + "\n")
    f.write(f"训练集: MAE={mae_train:.4f}, RMSE={rmse_train:.4f}, R²={r2_train:.4f}\n")
    f.write(f"测试集: MAE={mae_test:.4f}, RMSE={rmse_test:.4f}, R²={r2_test:.4f}\n")
print("最优参数已保存到: sarimax_best_params.txt")
