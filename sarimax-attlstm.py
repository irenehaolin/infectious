# === 1. 导入库 === 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from statsmodels.tsa.statespace.sarimax import SARIMAX 

import tensorflow as tf 
from tensorflow.keras.models import Model, save_model, load_model 
from tensorflow.keras.layers import Input, Dense, LSTM, Layer, Softmax 
from tensorflow.keras.optimizers import Adam 
import random 
import os 
import pickle 
import sys

# 设置随机种子 
SEED = 65431 
np.random.seed(SEED) 
tf.random.set_seed(SEED) 
random.seed(SEED) 
os.environ['PYTHONHASHSEED'] = str(SEED) 

# === 2. 读取并处理数据 === 
file_path = "data1_lagged_final_vars.xlsx"
if not os.path.exists(file_path):
    file_path = os.path.join(os.path.dirname(sys.argv[0]), file_path)
if not os.path.exists(file_path):
    raise FileNotFoundError(f"数据文件 {file_path} 未找到，请确保文件存在于当前目录或脚本目录")
data = pd.read_excel(file_path) 
data['Date'] = pd.to_datetime(data['Date']) 
data.set_index('Date', inplace=True) 

# 对 ILI% 进行对数变换
data['log_Influenza'] = np.log1p(data['Influenza']) 

# === 3. 初始化归一化器 === 
ili_scaler = MinMaxScaler()
temp_scaler = MinMaxScaler()
rain_scaler = MinMaxScaler()
x1_scaler = MinMaxScaler()
x2_scaler = MinMaxScaler()
x3_scaler = MinMaxScaler()
x4_scaler = MinMaxScaler()
x5_scaler = MinMaxScaler()
x7_scaler = MinMaxScaler()
vaccine_scaler = MinMaxScaler()

# === 4. 只在训练集上拟合归一化器，然后转换整个数据集 === 
test_size = 26
train_data_for_scaling = data.iloc[:-test_size].copy()

data_scaled = pd.DataFrame(index=data.index)

# 归一化所有变量，只使用训练集拟合
ili_scaler.fit(train_data_for_scaling[['log_Influenza']])
data_scaled['log_Influenza_scaled'] = ili_scaler.transform(data[['log_Influenza']])

temp_scaler.fit(train_data_for_scaling[['WAT']])
data_scaled['temp_scaled'] = temp_scaler.transform(data[['WAT']])

rain_scaler.fit(train_data_for_scaling[['WCR']])
data_scaled['rain_scaled'] = rain_scaler.transform(data[['WCR']])

x1_scaler.fit(train_data_for_scaling[['X1']])
data_scaled['X1_scaled'] = x1_scaler.transform(data[['X1']])

x2_scaler.fit(train_data_for_scaling[['X2']])
data_scaled['X2_scaled'] = x2_scaler.transform(data[['X2']])

x3_scaler.fit(train_data_for_scaling[['X3']])
data_scaled['X3_scaled'] = x3_scaler.transform(data[['X3']])

x4_scaler.fit(train_data_for_scaling[['X4']])
data_scaled['X4_scaled'] = x4_scaler.transform(data[['X4']])

x5_scaler.fit(train_data_for_scaling[['X5']])
data_scaled['X5_scaled'] = x5_scaler.transform(data[['X5']])

x7_scaler.fit(train_data_for_scaling[['X7']])
data_scaled['X7_scaled'] = x7_scaler.transform(data[['X7']])

vaccine_scaler.fit(train_data_for_scaling[['Vaccine']])
data_scaled['Vaccine_scaled'] = vaccine_scaler.transform(data[['Vaccine']])

data = pd.concat([data, data_scaled], axis=1)

# === 5. 拆分训练和测试集（包含归一化后的数据） === 
train_data = data.iloc[:-test_size]
test_data = data.iloc[-test_size:]

# === 6. 准备SARIMAX模型数据 === 
train_exog = train_data[['temp_scaled','rain_scaled','X1_scaled','X2_scaled','X3_scaled', 
                 'X4_scaled','X5_scaled','X7_scaled','Vaccine_scaled']] 
test_exog = test_data[['temp_scaled','rain_scaled','X1_scaled','X2_scaled','X3_scaled', 
                 'X4_scaled','X5_scaled','X7_scaled','Vaccine_scaled']] 

# === 6. SARIMAX建模 === 
sarimax_model = SARIMAX(train_data['log_Influenza_scaled'], 
                         exog=train_exog, 
                         order=(2, 0, 2), 
                         seasonal_order=(2, 0, 1, 52)) 
sarimax_model_fit = sarimax_model.fit(disp=False) 
print(sarimax_model_fit.summary()) 

# 训练集SARIMAX预测残差 
sarimax_train_pred = sarimax_model_fit.predict( 
     start=0, 
     end=len(train_data) - 1, 
     exog=train_exog 
 ) 

residuals = train_data['log_Influenza_scaled'].values - sarimax_train_pred 

# === 7. 构建 Attention-LSTM 输入序列 === 
def create_sequences(features, target, seq_len): 
    X, y = [], [] 
    for i in range(len(target) - seq_len): 
        X.append(features[i:i+seq_len]) 
        y.append(target[i+seq_len]) 
    return np.array(X), np.array(y) 

seq_len = 4 
res_features = train_data[['log_Influenza_scaled','temp_scaled','rain_scaled','X1_scaled','X2_scaled','X3_scaled', 
                 'X4_scaled','X5_scaled','X7_scaled','Vaccine_scaled']].values 
X_res, y_res = create_sequences(res_features, residuals, seq_len) 

all_features = data[['log_Influenza_scaled','temp_scaled','rain_scaled','X1_scaled','X2_scaled','X3_scaled', 
                 'X4_scaled','X5_scaled','X7_scaled','Vaccine_scaled']].values 
X_all_seq, _ = create_sequences(all_features, all_features[:, 0], seq_len) 
X_res_test = X_all_seq[-test_size:] 

# === 8. 构建 Attention-LSTM 模型 === 
class AttentionLayer(Layer): 
    def __init__(self, **kwargs): 
        super(AttentionLayer, self).__init__(**kwargs) 
        self.W = Dense(1, activation='tanh') 
        self.softmax = Softmax(axis=1) 

    def call(self, inputs): 
        score = self.W(inputs) 
        weights = self.softmax(score) 
        context = tf.reduce_sum(inputs * weights, axis=1) 
        return context 

input_layer = Input(shape=(seq_len, 10)) 
lstm_out = LSTM(104, return_sequences=True)(input_layer) 
attention_out = AttentionLayer()(lstm_out) 
output = Dense(1)(attention_out) 
res_model = Model(inputs=input_layer, outputs=output) 
res_model.compile(optimizer=Adam(0.001), loss='mse') 
res_model.summary() 

# === 9. 训练模型 === 
res_model.fit(X_res, y_res, epochs=400, batch_size=4, verbose=1) 

# === 10. 训练集预测（融合） === 
residual_train_pred = res_model.predict(X_res) 
residual_train_pred = residual_train_pred[:, 0] 

sarimax_train_pred_short = sarimax_model_fit.predict( 
     start=seq_len, 
     end=len(train_data) - 1, 
     exog=train_exog.iloc[seq_len:] 
 ) 

hybrid_train_pred_scaled = sarimax_train_pred_short + residual_train_pred 
hybrid_train_pred_log = ili_scaler.inverse_transform(hybrid_train_pred_scaled.values.reshape(-1, 1)) 
hybrid_train_pred = np.expm1(hybrid_train_pred_log) 

train_actual_log = ili_scaler.inverse_transform(train_data['log_Influenza_scaled'].values.reshape(-1, 1)[seq_len:]) 
train_actual = np.expm1(train_actual_log) 

mae_train = mean_absolute_error(train_actual, hybrid_train_pred) 
rmse_train = np.sqrt(mean_squared_error(train_actual, hybrid_train_pred)) 
r2_train = r2_score(train_actual, hybrid_train_pred) 

print(f"\n【训练集】SARIMAX-AttentionLSTM混合模型评估：\nMAE = {mae_train:.4f}, RMSE = {rmse_train:.4f}, R² = {r2_train:.4f}") 

# === 11. 测试集预测（融合） === 
residual_pred = res_model.predict(X_res_test) 
residual_pred = residual_pred[:, 0] 

train_len = len(train_data) 
total_len = len(data) 

sarimax_test_pred = sarimax_model_fit.predict( 
     start=train_len, 
     end=total_len - 1, 
     exog=test_exog 
 ) 

hybrid_pred_scaled = sarimax_test_pred + residual_pred 
hybrid_pred_log = ili_scaler.inverse_transform(hybrid_pred_scaled.values.reshape(-1, 1)) 
hybrid_pred = np.expm1(hybrid_pred_log) 

test_actual_log = ili_scaler.inverse_transform(test_data['log_Influenza_scaled'].values.reshape(-1, 1)) 
test_actual = np.expm1(test_actual_log) 

mae_test = mean_absolute_error(test_actual, hybrid_pred) 
rmse_test = np.sqrt(mean_squared_error(test_actual, hybrid_pred)) 
r2_test = r2_score(test_actual, hybrid_pred) 

print(f"\n【测试集】SARIMAX-AttentionLSTM混合模型评估：\nMAE = {mae_test:.4f}, RMSE = {rmse_test:.4f}, R² = {r2_test:.4f}") 

# === 12. 计算95%预测区间（使用Bootstrap方法） === 
def bootstrap_prediction_interval(y_true, y_pred, alpha=0.05, n_bootstraps=100):
    residuals = y_true.flatten() - y_pred.flatten()
    n = len(residuals)
    bootstrapped_intervals = []
    
    for _ in range(n_bootstraps):
        bootstrap_residuals = np.random.choice(residuals, size=n, replace=True)
        bootstrap_pred = y_pred.flatten() + bootstrap_residuals
        bootstrapped_intervals.append(bootstrap_pred)
    
    bootstrapped_intervals = np.array(bootstrapped_intervals)
    lower = np.percentile(bootstrapped_intervals, (alpha/2)*100, axis=0)
    upper = np.percentile(bootstrapped_intervals, (1 - alpha/2)*100, axis=0)
    
    return lower, upper

test_lower, test_upper = bootstrap_prediction_interval(test_actual, hybrid_pred, alpha=0.05, n_bootstraps=1000)
train_lower, train_upper = bootstrap_prediction_interval(train_actual, hybrid_train_pred, alpha=0.05, n_bootstraps=1000)

# === 图 1：训练集拟合趋势 ===
train_dates = train_data.index[seq_len:]
plt.figure(figsize=(12, 6))
plt.plot(train_dates, train_actual, label='Training Actual')
plt.plot(train_dates, hybrid_train_pred.flatten(), label='Training Fitted')
plt.title('Training Set Fitting SARIMAX-AttLSTM Model')
plt.xlabel('Time')
plt.ylabel('Influenza incidence')
plt.legend()
plt.grid()
plt.savefig('SARIMAX-AttLSTM_（全部）训练集12.18.pdf', format='pdf', dpi=300)
plt.close()

# === 图 2：测试集拟合趋势 + 95%预测区间 === 
plt.figure(figsize=(12, 6)) 
plt.plot(test_data.index, test_actual.flatten(), label='Actual') 
plt.plot(test_data.index, hybrid_pred.flatten(), label='Predicted', color='orange') 
plt.fill_between(test_data.index, test_lower, test_upper, 
                  color='gray', alpha=0.3, label='95% Prediction Interval') 

plt.title('Test Set Prediction with 95% Prediction Interval (SARIMAX-AttLSTM)') 
plt.xlabel('Time') 
plt.ylabel('Influenza incidence') 
plt.legend() 
plt.grid() 
plt.savefig('SARIMAX-AttLSTM_（全部）验证集12.18.pdf', format='pdf', dpi=300) 
plt.close()

# === 保存模型和归一化器 ===
model_dir = 'saved_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 保存SARIMAX模型
try:
    sarimax_model_fit.save(os.path.join(model_dir, 'sarimax_model.pkl'))
except:
    with open(os.path.join(model_dir, 'sarimax_model.pkl'), 'wb') as f:
        pickle.dump(sarimax_model_fit, f)

# 保存Attention-LSTM模型
save_model(res_model, os.path.join(model_dir, 'attlstm_model.keras'))#  保存所有归一化器
scalers = {
    'ili_scaler': ili_scaler,
    'temp_scaler': temp_scaler,
    'rain_scaler': rain_scaler,
    'x1_scaler': x1_scaler,
    'x2_scaler': x2_scaler,
    'x3_scaler': x3_scaler,
    'x4_scaler': x4_scaler,
    'x5_scaler': x5_scaler,
    'x7_scaler': x7_scaler,
    'vaccine_scaler': vaccine_scaler
}

with open(os.path.join(model_dir, 'scalers.pkl'), 'wb') as f:
    pickle.dump(scalers, f)

# 保存序列长度
with open(os.path.join(model_dir, 'seq_len.pkl'), 'wb') as f:
    pickle.dump(seq_len, f)

# 保存原始数据列名和相关信息
with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
    pickle.dump({
        'test_size': test_size,
        'seq_len': seq_len,
        'features_columns': ['log_Influenza_scaled','temp_scaled','rain_scaled','X1_scaled','X2_scaled','X3_scaled',
                  'X4_scaled','X5_scaled','X7_scaled','Vaccine_scaled'],
        'exog_columns': ['temp_scaled','rain_scaled','X1_scaled','X2_scaled','X3_scaled',
                  'X4_scaled','X5_scaled','X7_scaled','Vaccine_scaled']
    }, f)

print(f"模型和归一化器已成功保存到 {model_dir} 目录")
