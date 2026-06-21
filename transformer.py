"""
Transformer疾病发病率预测模型 - 带Bootstrap置信区间
====================================================
功能：
1. 使用Transformer模型预测疾病发病率
2. 使用Bootstrap方法计算95%置信区间
3. 支持多变量输入和时间序列特征
4. 可视化预测结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import random
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
def set_chinese_font():
    try:
        font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        return font
    except:
        print("警告：无法加载中文字体，将使用默认字体")
        return None

chinese_font = set_chinese_font()

# 设置随机种子
SEED = 1314
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


# ================================================================
# 1. 数据加载
# ================================================================
def load_data(file_path, target_column='Influenza'):
    """加载Excel数据文件"""
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(sys.argv[0]), file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件 {file_path} 未找到")

    data = pd.read_excel(file_path)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

    data['log_Target'] = np.log1p(data[target_column])
    return data


# ================================================================
# 2. 数据预处理
# ================================================================
class DataPreprocessor:
    """数据预处理类"""

    def __init__(self, seq_len=4):
        self.seq_len = seq_len
        self.scaler = MinMaxScaler()
        self.feat_scaler = MinMaxScaler()
        self.train_dates = None
        self.test_dates = None

    def prepare_data(self, data, test_size=26, feature_columns=None):
        """准备训练和测试数据"""
        train_data = data.iloc[:-test_size].copy()
        test_data = data.iloc[-test_size:].copy()

        # 归一化目标变量
        self.scaler.fit(train_data[['log_Target']])
        data['Target_scaled'] = self.scaler.transform(data[['log_Target']])

        # 准备特征（如果有额外特征）
        if feature_columns:
            self.feat_scaler.fit(train_data[feature_columns])
            scaled_features = self.feat_scaler.transform(data[feature_columns])
            features = np.concatenate([data['Target_scaled'].values.reshape(-1, 1), scaled_features], axis=1)
        else:
            features = data['Target_scaled'].values.reshape(-1, 1)

        # 创建序列
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
            y.append(features[i + seq_len, 0])  # 只预测目标变量
        return np.array(X), np.array(y)

    def inverse_transform(self, values):
        """反归一化"""
        return self.scaler.inverse_transform(values.reshape(-1, 1))

    def inverse_log(self, values):
        """反对数变换"""
        return np.expm1(values)


# ================================================================
# 3. Transformer模型构建
# ================================================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """Transformer编码器块"""
    # Multi-Head Attention
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Add()([x, inputs])  # Residual connection
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed Forward Network
    x = Dense(ff_dim, activation='relu')(x)
    x = Dense(inputs.shape[-1])(x)
    x = Add()([x, inputs])  # Residual connection
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(dropout)(x)

    return x


def build_transformer_model(seq_len, n_features, head_size=32, num_heads=4, 
                           ff_dim=64, num_layers=3, dropout=0.1):
    """
    构建Transformer时间序列预测模型

    参数:
        seq_len: 序列长度
        n_features: 特征数量
        head_size: 注意力头的维度
        num_heads: 注意力头数量
        ff_dim: 前馈网络维度
        num_layers: Transformer层数
        dropout: Dropout比例

    返回:
        model: 编译后的Keras模型
    """
    from tensorflow.keras.layers import GlobalAveragePooling1D
    
    inputs = Input(shape=(seq_len, n_features))
    x = inputs

    # 添加Transformer编码器层
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # 全局平均池化（使用Keras层）
    x = GlobalAveragePooling1D()(x)

    # 输出层
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


# ================================================================
# 4. Bootstrap置信区间计算
# ================================================================
def bootstrap_confidence_interval(y_true, y_pred, n_bootstraps=1000, alpha=0.05):
    """
    使用Bootstrap方法计算预测区间

    参数:
        y_true: 真实值
        y_pred: 预测值
        n_bootstraps: Bootstrap采样次数
        alpha: 显著性水平

    返回:
        lower: 预测区间下界
        upper: 预测区间上界
    """
    residuals = y_true - y_pred
    n = len(residuals)
    bootstrapped_predictions = []

    for _ in range(n_bootstraps):
        bootstrap_indices = np.random.choice(n, size=n, replace=True)
        bootstrap_residuals = residuals[bootstrap_indices]
        bootstrap_pred = y_pred + bootstrap_residuals
        bootstrapped_predictions.append(bootstrap_pred)

    bootstrapped_predictions = np.array(bootstrapped_predictions)

    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(bootstrapped_predictions, lower_percentile, axis=0)
    upper = np.percentile(bootstrapped_predictions, upper_percentile, axis=0)

    return lower, upper


# ================================================================
# 5. 模型训练和预测
# ================================================================
def train_and_predict(X_train, y_train, X_test, y_test, preprocessor,
                      epochs=100, batch_size=32, patience=10, verbose=1):
    """训练模型并进行预测"""
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]

    # 构建Transformer模型
    model = build_transformer_model(seq_len, n_features)
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
    print("\n开始训练Transformer模型...")
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
        'y_train': preprocessor.inverse_log(preprocessor.inverse_transform(y_train).flatten()),
        'y_train_pred': preprocessor.inverse_log(preprocessor.inverse_transform(y_train_pred_scaled).flatten()),
        'y_test': preprocessor.inverse_log(preprocessor.inverse_transform(y_test).flatten()),
        'y_test_pred': preprocessor.inverse_log(preprocessor.inverse_transform(y_test_pred_scaled).flatten()),
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
# 6. 评估指标计算
# ================================================================
def calculate_metrics(y_true, y_pred):
    """计算回归评估指标"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    }


# ================================================================
# 7. 可视化
# ================================================================
def plot_results(train_dates, test_dates, predictions, save_path=None, show_plots=True):
    """绘制预测结果"""
    fig = plt.figure(figsize=(20, 16))

    # 图1: 训练集拟合
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(train_dates, predictions['y_train'], label='实际值', color='#1f77b4', linewidth=1.5)
    ax1.plot(train_dates, predictions['y_train_pred'], label='拟合值', color='#ff7f0e', linewidth=1.5)
    ax1.set_title('训练集拟合效果', fontsize=14, fontweight='bold')
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('发病率', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # 图2: 测试集预测（带置信区间）
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(test_dates, predictions['y_test'], label='实际值', color='#1f77b4', linewidth=2)
    ax2.plot(test_dates, predictions['y_test_pred'], label='预测值', color='#ff7f0e', linewidth=2)
    ax2.fill_between(
        test_dates,
        predictions['lower'],
        predictions['upper'],
        color='gray', alpha=0.3, label='95%置信区间'
    )
    ax2.set_title('测试集预测效果（含95%置信区间）', fontsize=14, fontweight='bold')
    ax2.set_xlabel('时间', fontsize=12)
    ax2.set_ylabel('发病率', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # 图3: 训练损失曲线
    ax3 = fig.add_subplot(2, 2, 3)
    history = predictions['history']
    ax3.plot(history['loss'], label='训练损失', color='#1f77b4')
    ax3.plot(history['val_loss'], label='验证损失', color='#d62728')
    ax3.set_title('模型训练损失曲线', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss (MSE)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 图4: 预测误差分布
    ax4 = fig.add_subplot(2, 2, 4)
    errors = predictions['y_test'] - predictions['y_test_pred']
    ax4.hist(errors, bins=15, color='#1f77b4', edgecolor='black', alpha=0.7, density=True)
    ax4.axvline(x=0, color='#d62728', linestyle='--', linewidth=2, label='零误差线')
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax4.text(0.95, 0.95, f'均值: {mean_error:.4f}\n标准差: {std_error:.4f}', 
             transform=ax4.transAxes, ha='right', va='top', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    ax4.set_title('测试集预测误差分布', fontsize=14, fontweight='bold')
    ax4.set_xlabel('预测误差', fontsize=12)
    ax4.set_ylabel('频数/概率', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(pad=3.0)

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")

    if show_plots:
        plt.show()

    plt.close()


# ================================================================
# 8. 主程序
# ================================================================
def main():
    """主程序入口"""
    print("=" * 60)
    print("Transformer疾病发病率预测模型 - 带Bootstrap置信区间")
    print("=" * 60)

    # 配置参数
    FILE_PATH = "data1_lagged_final_vars.xlsx"
    TARGET_COLUMN = "Influenza"
    TEST_SIZE = 26
    SEQ_LEN = 4
    EPOCHS = 100
    BATCH_SIZE = 32
    PATIENCE = 10

    # 可选的额外特征
    FEATURE_COLUMNS = ['WAT', 'WCR', 'X1', 'X2', 'X3', 'X4', 'X5', 'X7', 'Vaccine']

    # 1. 加载数据
    print("\n[1] 加载数据...")
    data = load_data(FILE_PATH, TARGET_COLUMN)
    print(f"    数据形状: {data.shape}")
    print(f"    时间范围: {data.index.min()} 至 {data.index.max()}")

    # 2. 数据预处理
    print("\n[2] 数据预处理...")
    preprocessor = DataPreprocessor(seq_len=SEQ_LEN)
    
    # 检查是否存在额外特征列
    available_features = [col for col in FEATURE_COLUMNS if col in data.columns]
    if available_features:
        print(f"    使用额外特征: {available_features}")
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            data, TEST_SIZE, available_features
        )
    else:
        print("    未找到额外特征，使用单变量预测")
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(data, TEST_SIZE, None)
    
    print(f"    训练集: {X_train.shape[0]} 样本")
    print(f"    测试集: {X_test.shape[0]} 样本")
    print(f"    特征维度: {X_train.shape[2]}")

    # 3. 训练模型
    print("\n[3] 训练Transformer模型...")
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
    plot_results(
        preprocessor.train_dates,
        preprocessor.test_dates,
        predictions,
        save_path='transformer_prediction_results.pdf',
        show_plots=True
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
    results_df.to_csv('transformer_predictions.csv', index=False, encoding='utf-8-sig')
    print("    预测结果已保存到: transformer_predictions.csv")

    # 7. 保存模型
    print("\n[7] 保存模型...")
    model.save('transformer_disease_model.keras')
    print("    模型已保存到: transformer_disease_model.keras")

    # 保存预处理器
    import pickle
    with open('transformer_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("    预处理器已保存到: transformer_preprocessor.pkl")

    print("\n" + "=" * 60)
    print("程序执行完成！")
    print("=" * 60)

    return model, predictions, preprocessor


# ================================================================
# 9. 模型加载和预测
# ================================================================
def load_saved_model(model_path='transformer_disease_model.keras',
                     preprocessor_path='transformer_preprocessor.pkl'):
    """加载已保存的模型和预处理器"""
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
    """使用已有模型进行未来预测"""
    predictions = []
    current_sequence = recent_data[-preprocessor.seq_len:].copy()

    for _ in range(n_steps):
        X = current_sequence[-preprocessor.seq_len:].reshape(1, preprocessor.seq_len, -1)
        pred_scaled = model.predict(X, verbose=0)[0, 0]
        pred = preprocessor.inverse_log(preprocessor.inverse_transform(np.array([pred_scaled]))[0, 0])
        predictions.append(pred)
        current_sequence = np.append(current_sequence, pred_scaled)

    return np.array(predictions)


# 运行主程序
if __name__ == "__main__":
    main()
