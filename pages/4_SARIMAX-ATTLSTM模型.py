"""
SARIMAX-ATTLSTM混合模型预测页面
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="SARIMAX-ATTLSTM模型预测",
    page_icon="🎯",
    layout="wide"
)

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.title("🎯 SARIMAX-ATTLSTM混合模型预测")
st.markdown("---")

# 侧边栏参数设置
st.sidebar.header("模型参数设置")

st.sidebar.subheader("SARIMAX参数")
p = st.sidebar.slider("p (AR阶数)", min_value=0, max_value=3, value=2)
d = st.sidebar.slider("d (差分阶数)", min_value=0, max_value=2, value=0)
q = st.sidebar.slider("q (MA阶数)", min_value=0, max_value=3, value=2)
P = st.sidebar.slider("P (季节性AR阶数)", min_value=0, max_value=3, value=2)
D = st.sidebar.slider("D (季节性差分阶数)", min_value=0, max_value=2, value=0)
Q = st.sidebar.slider("Q (季节性MA阶数)", min_value=0, max_value=3, value=1)
s = st.sidebar.slider("s (季节周期)", min_value=4, max_value=52, value=52)

st.sidebar.subheader("Attention-LSTM参数")
seq_len = st.sidebar.slider("序列长度", min_value=2, max_value=12, value=4)
epochs = st.sidebar.slider("训练轮数", min_value=50, max_value=500, value=400)
batch_size = st.sidebar.slider("批次大小", min_value=4, max_value=32, value=4)
lstm_units = st.sidebar.slider("LSTM单元数", min_value=32, max_value=128, value=104)
learning_rate = st.sidebar.number_input("学习率", value=0.001, min_value=0.0001, max_value=0.01, format="%.4f")

st.sidebar.subheader("其他参数")
test_size = st.sidebar.slider("测试集大小", min_value=10, max_value=52, value=26)
seed = st.sidebar.number_input("随机种子", value=65431, min_value=1)

# 主页面内容
st.header("数据上传")

uploaded_file = st.file_uploader("上传Excel数据文件（需包含多变量特征列）", type=['xlsx', 'xls'])

if uploaded_file is not None:
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # 加载并显示数据
        data = pd.read_excel(tmp_path)
        
        st.subheader("数据预览")
        st.dataframe(data.head(10))
        
        # 检查必需列
        required_cols = ['Date', 'Influenza']
        optional_cols = ['WAT', 'WCR', 'X1', 'X2', 'X3', 'X4', 'X5', 'X7', 'Vaccine']
        
        missing_required = [col for col in required_cols if col not in data.columns]
        available_optional = [col for col in optional_cols if col in data.columns]
        
        if missing_required:
            st.error(f"缺少必需列: {missing_required}")
        else:
            st.success("数据格式验证通过！")
            
            if available_optional:
                st.info(f"检测到特征列: {available_optional}")
            else:
                st.warning("未检测到特征列，将使用单变量预测")
            
            # 显示数据统计
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("数据行数", len(data))
            with col2:
                st.metric("数据列数", len(data.columns))
            with col3:
                st.metric("时间范围", f"{data['Date'].min()} 至 {data['Date'].max()}")
            
            st.markdown("---")
            
            # 开始训练按钮
            if st.button("🚀 开始训练模型", type="primary"):
                # 设置随机种子
                import tensorflow as tf
                import random
                np.random.seed(seed)
                tf.random.set_seed(seed)
                random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    from sklearn.preprocessing import MinMaxScaler
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    from tensorflow.keras.models import Model
                    from tensorflow.keras.layers import Input, Dense, LSTM, Layer, Softmax
                    from tensorflow.keras.optimizers import Adam
                    
                    status_text.text("正在加载数据...")
                    progress_bar.progress(5)
                    
                    # 数据处理
                    data_processed = data.copy()
                    data_processed['Date'] = pd.to_datetime(data_processed['Date'])
                    data_processed.set_index('Date', inplace=True)
                    
                    # log变换
                    data_processed['log_Influenza'] = np.log1p(data_processed['Influenza'])
                    
                    # 初始化归一化器
                    scalers = {}
                    columns_to_scale = ['log_Influenza'] + available_optional
                    
                    train_data_for_scaling = data_processed.iloc[:-test_size].copy()
                    
                    for col in columns_to_scale:
                        scaler = MinMaxScaler()
                        scaler.fit(train_data_for_scaling[[col]])
                        data_processed[f'{col}_scaled'] = scaler.transform(data_processed[[col]])
                        scalers[col] = scaler
                    
                    progress_bar.progress(10)
                    
                    # 拆分训练和测试集
                    train_data = data_processed.iloc[:-test_size]
                    test_data = data_processed.iloc[-test_size:]
                    
                    status_text.text("正在构建SARIMAX模型...")
                    progress_bar.progress(15)
                    
                    # 准备外生变量
                    if available_optional:
                        exog_cols = [f'{col}_scaled' for col in available_optional]
                        train_exog = train_data[exog_cols]
                        test_exog = test_data[exog_cols]
                    else:
                        exog_cols = None
                        train_exog = None
                        test_exog = None
                    
                    # SARIMAX建模
                    sarimax_model = SARIMAX(
                        train_data['log_Influenza_scaled'],
                        exog=train_exog,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s)
                    )
                    
                    status_text.text("正在拟合SARIMAX模型...")
                    progress_bar.progress(25)
                    
                    sarimax_fit = sarimax_model.fit(disp=False)
                    
                    # 训练集SARIMAX预测残差
                    sarimax_train_pred = sarimax_fit.predict(
                        start=0, end=len(train_data) - 1, exog=train_exog
                    )
                    
                    residuals = train_data['log_Influenza_scaled'].values - sarimax_train_pred
                    
                    progress_bar.progress(35)
                    
                    status_text.text("正在准备Attention-LSTM数据...")
                    
                    # 构建Attention-LSTM输入序列
                    def create_sequences(features, target, seq_len):
                        X, y = [], []
                        for i in range(len(target) - seq_len):
                            X.append(features[i:i+seq_len])
                            y.append(target[i+seq_len])
                        return np.array(X), np.array(y)
                    
                    # 准备特征
                    feature_cols = ['log_Influenza_scaled'] + (exog_cols if exog_cols else [])
                    n_features = len(feature_cols)
                    
                    res_features = train_data[feature_cols].values
                    X_res, y_res = create_sequences(res_features, residuals, seq_len)
                    
                    progress_bar.progress(40)
                    
                    status_text.text("正在构建Attention-LSTM模型...")
                    
                    # Attention层
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
                    
                    # 构建模型
                    input_layer = Input(shape=(seq_len, n_features))
                    lstm_out = LSTM(lstm_units, return_sequences=True)(input_layer)
                    attention_out = AttentionLayer()(lstm_out)
                    output = Dense(1)(attention_out)
                    res_model = Model(inputs=input_layer, outputs=output)
                    res_model.compile(optimizer=Adam(learning_rate), loss='mse')
                    
                    # 显示模型结构
                    st.subheader("Attention-LSTM模型结构")
                    model_summary = []
                    res_model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text("\n".join(model_summary))
                    
                    progress_bar.progress(45)
                    
                    status_text.text("正在训练Attention-LSTM模型...")
                    
                    # 训练模型
                    history = res_model.fit(X_res, y_res, epochs=epochs, batch_size=batch_size, verbose=0)
                    
                    progress_bar.progress(80)
                    
                    status_text.text("正在进行预测...")
                    
                    # 训练集预测（融合）
                    residual_train_pred = res_model.predict(X_res, verbose=0)[:, 0]
                    
                    sarimax_train_pred_short = sarimax_fit.predict(
                        start=seq_len, end=len(train_data) - 1, exog=train_exog.iloc[seq_len:] if train_exog is not None else None
                    )
                    
                    hybrid_train_pred_scaled = sarimax_train_pred_short + residual_train_pred
                    hybrid_train_pred_log = scalers['log_Influenza'].inverse_transform(
                        hybrid_train_pred_scaled.values.reshape(-1, 1)
                    )
                    hybrid_train_pred = np.expm1(hybrid_train_pred_log)
                    
                    train_actual_log = scalers['log_Influenza'].inverse_transform(
                        train_data['log_Influenza_scaled'].values.reshape(-1, 1)[seq_len:]
                    )
                    train_actual = np.expm1(train_actual_log)
                    
                    # 测试集预测（融合）
                    all_features = data_processed[feature_cols].values
                    X_all_seq, _ = create_sequences(all_features, all_features[:, 0], seq_len)
                    X_res_test = X_all_seq[-test_size:]
                    
                    residual_pred = res_model.predict(X_res_test, verbose=0)[:, 0]
                    
                    sarimax_test_pred = sarimax_fit.predict(
                        start=len(train_data), end=len(data_processed) - 1, exog=test_exog
                    )
                    
                    hybrid_pred_scaled = sarimax_test_pred + residual_pred
                    hybrid_pred_log = scalers['log_Influenza'].inverse_transform(
                        hybrid_pred_scaled.values.reshape(-1, 1)
                    )
                    hybrid_pred = np.expm1(hybrid_pred_log)
                    
                    test_actual_log = scalers['log_Influenza'].inverse_transform(
                        test_data['log_Influenza_scaled'].values.reshape(-1, 1)
                    )
                    test_actual = np.expm1(test_actual_log)
                    
                    progress_bar.progress(90)
                    
                    # Bootstrap预测区间
                    def bootstrap_prediction_interval(y_true, y_pred, alpha=0.05, n_bootstraps=1000):
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
                    
                    test_lower, test_upper = bootstrap_prediction_interval(test_actual, hybrid_pred)
                    
                    progress_bar.progress(100)
                    status_text.text("训练完成！")
                    
                    # 计算评估指标
                    mae_train = mean_absolute_error(train_actual, hybrid_train_pred)
                    rmse_train = np.sqrt(mean_squared_error(train_actual, hybrid_train_pred))
                    r2_train = r2_score(train_actual, hybrid_train_pred)
                    
                    mae_test = mean_absolute_error(test_actual, hybrid_pred)
                    rmse_test = np.sqrt(mean_squared_error(test_actual, hybrid_pred))
                    r2_test = r2_score(test_actual, hybrid_pred)
                    
                    # 显示评估指标
                    st.markdown("---")
                    st.header("模型评估结果")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("训练集指标")
                        st.metric("MAE", f"{mae_train:.4f}")
                        st.metric("RMSE", f"{rmse_train:.4f}")
                        st.metric("R²", f"{r2_train:.4f}")
                    
                    with col2:
                        st.subheader("测试集指标")
                        st.metric("MAE", f"{mae_test:.4f}")
                        st.metric("RMSE", f"{rmse_test:.4f}")
                        st.metric("R²", f"{r2_test:.4f}")
                    
                    # 可视化结果 - 训练集效果
                    st.markdown("---")
                    st.header("训练集拟合效果")
                    
                    train_dates = train_data.index[seq_len:]
                    fig_train, ax_train = plt.subplots(figsize=(14, 6))
                    ax_train.plot(train_dates, train_actual.flatten(), label='实际值', color='blue', linewidth=2)
                    ax_train.plot(train_dates, hybrid_train_pred.flatten(), label='拟合值', color='red', linewidth=2, alpha=0.8)
                    ax_train.set_title('Influenza - 训练集拟合效果 (SARIMAX-AttLSTM)', fontsize=14, fontweight='bold')
                    ax_train.set_xlabel('时间', fontsize=12)
                    ax_train.set_ylabel('发病率', fontsize=12)
                    ax_train.legend(fontsize=12)
                    ax_train.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig_train)
                    
                    # 可视化结果 - 测试集预测效果
                    st.markdown("---")
                    st.header("测试集预测效果")
                    
                    fig_test, ax_test = plt.subplots(figsize=(14, 6))
                    ax_test.plot(test_data.index, test_actual.flatten(), label='实际值', color='blue', linewidth=2)
                    ax_test.plot(test_data.index, hybrid_pred.flatten(), label='预测值', color='orange', linewidth=2)
                    ax_test.fill_between(test_data.index, test_lower, test_upper,
                                         color='gray', alpha=0.3, label='95% 预测区间')
                    ax_test.set_title('Influenza - 测试集预测效果（含95%置信区间）', fontsize=14, fontweight='bold')
                    ax_test.set_xlabel('时间', fontsize=12)
                    ax_test.set_ylabel('发病率', fontsize=12)
                    ax_test.legend(fontsize=12)
                    ax_test.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig_test)
                    
                    # 训练损失曲线
                    st.markdown("---")
                    st.header("训练损失曲线")
                    
                    fig_loss, ax_loss = plt.subplots(figsize=(14, 6))
                    ax_loss.plot(history.history['loss'], label='训练损失', color='blue', linewidth=2)
                    ax_loss.set_title('Attention-LSTM训练损失曲线', fontsize=14, fontweight='bold')
                    ax_loss.set_xlabel('Epoch', fontsize=12)
                    ax_loss.set_ylabel('Loss (MSE)', fontsize=12)
                    ax_loss.legend(fontsize=12)
                    ax_loss.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_loss)
                    
                    # 下载功能
                    st.markdown("---")
                    st.header("结果下载")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        buf_train = BytesIO()
                        fig_train.savefig(buf_train, format='pdf', dpi=300, bbox_inches='tight')
                        buf_train.seek(0)
                        st.download_button(
                            label="📥 下载训练集图表 (PDF)",
                            data=buf_train,
                            file_name="sarimax_attlstm_train_results.pdf",
                            mime="application/pdf"
                        )
                    
                    with col2:
                        buf_test = BytesIO()
                        fig_test.savefig(buf_test, format='pdf', dpi=300, bbox_inches='tight')
                        buf_test.seek(0)
                        st.download_button(
                            label="📥 下载测试集图表 (PDF)",
                            data=buf_test,
                            file_name="sarimax_attlstm_test_results.pdf",
                            mime="application/pdf"
                        )
                    
                    with col3:
                        results_df = pd.DataFrame({
                            'Date': test_data.index,
                            'Actual': test_actual.flatten(),
                            'Predicted': hybrid_pred.flatten(),
                            'Lower_95CI': test_lower,
                            'Upper_95CI': test_upper,
                            'Error': test_actual.flatten() - hybrid_pred.flatten()
                        })
                        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 下载预测结果 (CSV)",
                            data=csv,
                            file_name="sarimax_attlstm_predictions.csv",
                            mime="text/csv"
                        )
                    
                    plt.close('all')
                    
                except Exception as e:
                    st.error(f"训练过程中出现错误: {str(e)}")
                    st.exception(e)
                
    except Exception as e:
        st.error(f"数据加载错误: {str(e)}")

else:
    st.info("请上传包含多变量特征列的Excel数据文件开始预测")

# 清理临时文件
if uploaded_file is not None and 'tmp_path' in locals():
    try:
        os.unlink(tmp_path)
    except:
        pass