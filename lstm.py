"""
LSTM疾病发病率预测模型 - 带Bootstrap置信区间
=============================================
功能：
1. 使用LSTM模型预测疾病发病率
2. 使用Bootstrap方法计算95%置信区间
3. 数据预处理和模型训练
4. 可视化预测结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import random
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. 设置随机种子（确保结果可复现）
# ================================================================
SEED = 1314
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


# ================================================================
# 2. 数据加载
# ================================================================
def load_data(file_path, target_column='Influenza'):
    """
    加载Excel数据文件

    参数:
        file_path: 数据文件路径
        target_column: 目标变量列名（默认：Influenza）

    返回:
        data: 处理后的数据框
    """
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(sys.argv[0]), file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件 {file_path} 未找到")

    data = pd.read_excel(file_path)

    # 检查是否包含Date列
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

    # 对目标变量进行对数变换（稳定方差）
    data['log_Target'] = np.log1p(data[target_column])

    return data


# ================================================================
# 3. 数据预处理
# ================================================================
class DataPreprocessor:
    """数据预处理类"""

    def __init__(self, seq_len=4):
        self.seq_len = seq_len
        self.scaler = MinMaxScaler()
        self.train_dates = None
        self.test_dates = None

    def prepare_data(self, data, test_size=26):
        """
        准备训练和测试数据

        参数:
            data: 原始数据框
            test_size: 测试集样本数（默认：26）

        返回:
            X_train, y_train, X_test, y_test: 训练和测试数据
        """
        # 拆分训练和测试集
        train_data = data.iloc[:-test_size].copy()
        test_data = data.iloc[-test_size:].copy()

        # 使用训练集拟合归一化器，然后转换整个数据集
        self.scaler.fit(train_data[['log_Target']])
        data['Target_scaled'] = self.scaler.transform(data[['log_Target']])

        # 创建序列
        features = data['Target_scaled'].values
        X, y = self._create_sequences(features, self.seq_len)

        # 拆分训练和测试
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]

        # 保存日期索引
        self.train_dates = data.index[self.seq_len:self.seq_len + len(X_train)]
        self.test_dates = data.index[self.seq_len + len(X_train):self.seq_len + len(X)]

        return X_train, X_test, y_train, y_test

    def _create_sequences(self, features, seq_len):
        """创建时间序列样本"""
        X, y = [], []
        for i in range(len(features) - seq_len):
            X.append(features[i:i + seq_len])
            y.append(features[i + seq_len])
        return np.array(X), np.array(y)

    def inverse_transform(self, values):
        """反归一化"""
        return self.scaler.inverse_transform(values.reshape(-1, 1))

    def inverse_log(self, values):
        """反对数变换"""
        return np.expm1(values)


# ================================================================
# 4. LSTM模型构建
# ================================================================
def build_lstm_model(seq_len, lstm_units=64, dropout_rate=0.2, learning_rate=0.001):
    """
    构建LSTM模型

    参数:
        seq_len: 序列长度
        lstm_units: LSTM单元数（默认：64）
        dropout_rate: Dropout比例（默认：0.2）
        learning_rate: 学习率（默认：0.001）

    返回:
        model: 编译后的Keras模型
    """
    model = Sequential([
        LSTM(lstm_units, return_sequences=False, input_shape=(seq_len, 1)),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model


# ================================================================
# 5. Bootstrap置信区间计算
# ================================================================
def bootstrap_confidence_interval(y_true, y_pred, n_bootstraps=1000, alpha=0.05):
    """
    使用Bootstrap方法计算预测区间

    参数:
        y_true: 真实值
        y_pred: 预测值
        n_bootstraps: Bootstrap采样次数（默认：1000）
        alpha: 显著性水平（默认：0.05，即95%置信区间）

    返回:
        lower: 预测区间下界
        upper: 预测区间上界
    """
    residuals = y_true - y_pred
    n = len(residuals)
    bootstrapped_predictions = []

    for _ in range(n_bootstraps):
        # 有放回抽样
        bootstrap_indices = np.random.choice(n, size=n, replace=True)
        bootstrap_residuals = residuals[bootstrap_indices]
        # 生成Bootstrap预测
        bootstrap_pred = y_pred + bootstrap_residuals
        bootstrapped_predictions.append(bootstrap_pred)

    bootstrapped_predictions = np.array(bootstrapped_predictions)

    # 计算分位数
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(bootstrapped_predictions, lower_percentile, axis=0)
    upper = np.percentile(bootstrapped_predictions, upper_percentile, axis=0)

    return lower, upper


# ================================================================
# 6. 模型训练和预测
# ================================================================
def train_and_predict(X_train, y_train, X_test, y_test,
                      preprocessor, epochs=100, batch_size=32,
                      patience=10, verbose=1):
    """
    训练模型并进行预测

    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        preprocessor: 数据预处理器
        epochs: 训练轮数（默认：100）
        batch_size: 批次大小（默认：32）
        patience: 早停耐心值（默认：10）
        verbose: 训练详细程度（默认：1）

    返回:
        model: 训练好的模型
        predictions: 预测结果字典
    """
    seq_len = X_train.shape[1]

    # 构建模型
    model = build_lstm_model(seq_len)
    print("\n模型结构:")
    model.summary()

    # 早停回调
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )

    # 训练模型
    print("\n开始训练模型...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )

    # 训练集预测
    y_train_pred_scaled = model.predict(X_train, verbose=0)

    # 测试集预测
    y_test_pred_scaled = model.predict(X_test, verbose=0)

    # 反归一化和反对数变换
    predictions = {
        # 训练集
        'y_train': preprocessor.inverse_log(preprocessor.inverse_transform(y_train.reshape(-1, 1)).flatten()),
        'y_train_pred': preprocessor.inverse_log(preprocessor.inverse_transform(y_train_pred_scaled).flatten()),
        # 测试集
        'y_test': preprocessor.inverse_log(preprocessor.inverse_transform(y_test.reshape(-1, 1)).flatten()),
        'y_test_pred': preprocessor.inverse_log(preprocessor.inverse_transform(y_test_pred_scaled).flatten()),
        # 训练历史
        'history': history.history
    }

    # Bootstrap置信区间
    lower, upper = bootstrap_confidence_interval(
        predictions['y_test'],
        predictions['y_test_pred'],
        n_bootstraps=1000,
        alpha=0.05
    )
    predictions['lower'] = lower
    predictions['upper'] = upper

    return model, predictions


# ================================================================
# 7. 评估指标计算
# ================================================================
def calculate_metrics(y_true, y_pred):
    """计算回归评估指标"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }


# ================================================================
# 8. 可视化
# ================================================================
def plot_predictions(train_dates, test_dates, predictions, save_path=None):
    """
    绘制预测结果

    参数:
        train_dates: 训练集日期
        test_dates: 测试集日期
        predictions: 预测结果字典
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1: 训练集拟合
    ax1 = axes[0, 0]
    ax1.plot(train_dates, predictions['y_train'], label='实际值', color='blue')
    ax1.plot(train_dates, predictions['y_train_pred'], label='拟合值', color='red', alpha=0.7)
    ax1.set_title('训练集拟合效果', fontsize=12)
    ax1.set_xlabel('时间')
    ax1.set_ylabel('发病率')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 测试集预测（带置信区间）
    ax2 = axes[0, 1]
    ax2.plot(test_dates, predictions['y_test'], label='实际值', color='blue', linewidth=2)
    ax2.plot(test_dates, predictions['y_test_pred'], label='预测值', color='orange', linewidth=2)
    ax2.fill_between(
        test_dates,
        predictions['lower'],
        predictions['upper'],
        color='gray', alpha=0.3, label='95% 置信区间'
    )
    ax2.set_title('测试集预测效果（含95%置信区间）', fontsize=12)
    ax2.set_xlabel('时间')
    ax2.set_ylabel('发病率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 训练损失曲线
    ax3 = axes[1, 0]
    history = predictions['history']
    ax3.plot(history['loss'], label='训练损失', color='blue')
    ax3.plot(history['val_loss'], label='验证损失', color='red')
    ax3.set_title('模型训练损失曲线', fontsize=12)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (MSE)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4: 预测误差分布
    ax4 = axes[1, 1]
    errors = predictions['y_test'] - predictions['y_test_pred']
    ax4.hist(errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', label='零误差线')
    ax4.set_title('测试集预测误差分布', fontsize=12)
    ax4.set_xlabel('预测误差')
    ax4.set_ylabel('频数')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"图表已保存到: {save_path}")

    plt.close()


# ================================================================
# 9. 主程序
# ================================================================
def main():
    """主程序入口"""
    print("=" * 60)
    print("LSTM疾病发病率预测模型 - 带Bootstrap置信区间")
    print("=" * 60)

    # 配置参数
    FILE_PATH = "data1_lagged_final_vars.xlsx"  # 数据文件路径
    TARGET_COLUMN = "Influenza"  # 目标变量列名
    TEST_SIZE = 26  # 测试集大小
    SEQ_LEN = 4  # 序列长度
    EPOCHS = 300  # 训练轮数
    BATCH_SIZE = 32  # 批次大小
    PATIENCE = 10  # 早停耐心值
    LSTM_UNITS = 64  # LSTM单元数

    # 1. 加载数据
    print("\n[1] 加载数据...")
    data = load_data(FILE_PATH, TARGET_COLUMN)
    print(f"    数据形状: {data.shape}")
    print(f"    时间范围: {data.index.min()} 至 {data.index.max()}")

    # 2. 数据预处理
    print("\n[2] 数据预处理...")
    preprocessor = DataPreprocessor(seq_len=SEQ_LEN)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(data, TEST_SIZE)
    print(f"    训练集: {X_train.shape[0]} 样本, 特征维度: {X_train.shape[1]}")
    print(f"    测试集: {X_test.shape[0]} 样本")

    # 3. 训练模型
    print("\n[3] 训练LSTM模型...")
    model, predictions = train_and_predict(
        X_train, y_train, X_test, y_test,
        preprocessor,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE
    )

    # 4. 评估模型
    print("\n[4] 模型评估结果:")
    print("-" * 40)
    print("训练集指标:")
    train_metrics = calculate_metrics(predictions['y_train'], predictions['y_train_pred'])
    for metric, value in train_metrics.items():
        print(f"    {metric}: {value:.4f}")

    print("\n测试集指标:")
    test_metrics = calculate_metrics(predictions['y_test'], predictions['y_test_pred'])
    for metric, value in test_metrics.items():
        print(f"    {metric}: {value:.4f}")

    # 5. 绘制预测结果
    print("\n[5] 绘制预测结果...")
    plot_predictions(
        preprocessor.train_dates,
        preprocessor.test_dates,
        predictions,
        save_path='lstm_prediction_results.pdf'
    )

    # 6. 保存预测结果
    print("\n[6] 保存预测结果...")
    results_df = pd.DataFrame({
        'Date': preprocessor.test_dates,
        'Actual': predictions['y_test'],
        'Predicted': predictions['y_test_pred'],
        'Lower_95CI': predictions['lower'],
        'Upper_95CI': predictions['upper'],
        'Error': predictions['y_test'] - predictions['y_test_pred']
    })
    results_df.to_csv('lstm_predictions.csv', index=False, encoding='utf-8-sig')
    print("    预测结果已保存到: lstm_predictions.csv")

    # 7. 保存模型
    print("\n[7] 保存模型...")
    model.save('lstm_disease_model.keras')
    print("    模型已保存到: lstm_disease_model.keras")

    # 保存预处理器
    import pickle
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("    预处理器已保存到: preprocessor.pkl")

    print("\n" + "=" * 60)
    print("程序执行完成！")
    print("=" * 60)

    return model, predictions, preprocessor


# ================================================================
# 10. 使用示例和Streamlit部署入口
# ================================================================
def load_saved_model(model_path='lstm_disease_model.keras',
                     preprocessor_path='preprocessor.pkl'):
    """
    加载已保存的模型和预处理器

    参数:
        model_path: 模型文件路径
        preprocessor_path: 预处理器文件路径

    返回:
        model: 加载的模型
        preprocessor: 加载的预处理器
    """
    import pickle

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )

    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


def predict_future(model, preprocessor, recent_data, n_steps=4):
    """
    使用已有模型进行未来预测

    参数:
        model: 训练好的LSTM模型
        preprocessor: 数据预处理器
        recent_data: 最近的数据点（需要至少seq_len个点）
        n_steps: 预测步数

    返回:
        predictions: 未来预测值
    """
    predictions = []
    current_sequence = recent_data[-preprocessor.seq_len:].copy()

    for _ in range(n_steps):
        # 归一化
        scaled_seq = preprocessor.scaler.transform(current_sequence.reshape(-1, 1))
        X = scaled_seq[-preprocessor.seq_len:].reshape(1, preprocessor.seq_len, 1)

        # 预测
        pred_scaled = model.predict(X, verbose=0)[0, 0]

        # 反归一化和反对数
        pred = preprocessor.inverse_log(
            preprocessor.inverse_transform(np.array([pred_scaled]))[0, 0]
        )
        predictions.append(pred)

        # 更新序列
        current_sequence = np.append(current_sequence, pred_scaled)

    return np.array(predictions)


# 运行主程序
if __name__ == "__main__":
    main()
