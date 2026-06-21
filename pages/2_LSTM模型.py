"""
LSTM模型预测页面
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import os
import sys

# 设置页面配置
st.set_page_config(
    page_title="LSTM模型预测",
    page_icon="🔮",
    layout="wide"
)

# 导入模型模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lstm import (
    load_data, DataPreprocessor, build_lstm_model, 
    train_and_predict, calculate_metrics, bootstrap_confidence_interval
)

import tensorflow as tf
import random
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

st.title("🔮 LSTM模型预测")
st.markdown("---")

# 侧边栏参数设置
st.sidebar.header("模型参数设置")

seed = st.sidebar.number_input("随机种子", value=1314, min_value=1)
seq_len = st.sidebar.slider("序列长度", min_value=2, max_value=12, value=4)
epochs = st.sidebar.slider("训练轮数", min_value=50, max_value=500, value=100)
batch_size = st.sidebar.slider("批次大小", min_value=8, max_value=64, value=32)
lstm_units = st.sidebar.slider("LSTM单元数", min_value=32, max_value=128, value=64)
dropout_rate = st.sidebar.slider("Dropout比例", min_value=0.0, max_value=0.5, value=0.2)
learning_rate = st.sidebar.number_input("学习率", value=0.001, min_value=0.0001, max_value=0.01, format="%.4f")
test_size = st.sidebar.slider("测试集大小", min_value=10, max_value=52, value=26)
patience = st.sidebar.slider("早停耐心值", min_value=5, max_value=30, value=10)

# 主页面内容
st.header("数据上传")

uploaded_file = st.file_uploader("上传Excel数据文件", type=['xlsx', 'xls'])

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
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            st.error(f"缺少必需列: {missing_cols}")
        else:
            st.success("数据格式验证通过！")
            
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
                set_seed(seed)
                
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("正在加载数据...")
                    progress_bar.progress(10)
                    
                    # 加载数据
                    data_processed = load_data(tmp_path, 'Influenza')
                    
                    status_text.text("正在预处理数据...")
                    progress_bar.progress(20)
                    
                    # 数据预处理
                    preprocessor = DataPreprocessor(seq_len=seq_len)
                    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
                        data_processed, test_size=test_size
                    )
                    
                    status_text.text("正在构建LSTM模型...")
                    progress_bar.progress(30)
                    
                    # 构建模型
                    model = build_lstm_model(
                        seq_len, 
                        lstm_units=lstm_units,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate
                    )
                    
                    # 显示模型结构
                    st.subheader("模型结构")
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text("\n".join(model_summary))
                    
                    status_text.text("正在训练模型...")
                    progress_bar.progress(40)
                    
                    # 训练模型（使用回调更新进度）
                    from tensorflow.keras.callbacks import EarlyStopping, Callback
                    
                    class StreamlitProgressCallback(Callback):
                        def __init__(self, total_epochs, progress_bar, status_text):
                            super().__init__()
                            self.total_epochs = total_epochs
                            self.progress_bar = progress_bar
                            self.status_text = status_text
                            self.current_progress = 40
                        
                        def on_epoch_end(self, epoch, logs=None):
                            progress = 40 + int((epoch / self.total_epochs) * 50)
                            self.progress_bar.progress(min(progress, 90))
                            self.status_text.text(f"训练轮次 {epoch+1}/{self.total_epochs} - Loss: {logs.get('loss', 0):.4f}")
                    
                    early_stop = EarlyStopping(
                        monitor='val_loss',
                        patience=patience,
                        restore_best_weights=True,
                        verbose=0
                    )
                    
                    progress_callback = StreamlitProgressCallback(epochs, progress_bar, status_text)
                    
                    history = model.fit(
                        X_train, y_train,
                        validation_split=0.1,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stop, progress_callback],
                        verbose=0
                    )
                    
                    status_text.text("正在进行预测...")
                    progress_bar.progress(95)
                    
                    # 预测
                    y_train_pred_scaled = model.predict(X_train, verbose=0)
                    y_test_pred_scaled = model.predict(X_test, verbose=0)
                    
                    # 反归一化和反对数变换
                    y_train_actual = preprocessor.inverse_log(
                        preprocessor.inverse_transform(y_train.reshape(-1, 1)).flatten()
                    )
                    y_train_pred = preprocessor.inverse_log(
                        preprocessor.inverse_transform(y_train_pred_scaled).flatten()
                    )
                    y_test_actual = preprocessor.inverse_log(
                        preprocessor.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    )
                    y_test_pred = preprocessor.inverse_log(
                        preprocessor.inverse_transform(y_test_pred_scaled).flatten()
                    )
                    
                    # Bootstrap置信区间
                    lower, upper = bootstrap_confidence_interval(
                        y_test_actual, y_test_pred, n_bootstraps=1000, alpha=0.05
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("训练完成！")
                    
                    # 计算评估指标
                    train_metrics = calculate_metrics(y_train_actual, y_train_pred)
                    test_metrics = calculate_metrics(y_test_actual, y_test_pred)
                    
                    # 显示评估指标
                    st.markdown("---")
                    st.header("模型评估结果")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("训练集指标")
                        for metric, value in train_metrics.items():
                            st.metric(metric, f"{value:.4f}")
                    
                    with col2:
                        st.subheader("测试集指标")
                        for metric, value in test_metrics.items():
                            st.metric(metric, f"{value:.4f}")
                    
                    # 可视化结果 - 训练集效果
                    st.markdown("---")
                    st.header("训练集拟合效果")
                    
                    fig_train, ax_train = plt.subplots(figsize=(14, 6))
                    ax_train.plot(preprocessor.train_dates, y_train_actual, label='实际值', color='blue', linewidth=2)
                    ax_train.plot(preprocessor.train_dates, y_train_pred, label='拟合值', color='red', linewidth=2, alpha=0.8)
                    ax_train.set_title('Influenza - 训练集拟合效果', fontsize=14, fontweight='bold')
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
                    ax_test.plot(preprocessor.test_dates, y_test_actual, label='实际值', color='blue', linewidth=2)
                    ax_test.plot(preprocessor.test_dates, y_test_pred, label='预测值', color='orange', linewidth=2)
                    ax_test.fill_between(
                        preprocessor.test_dates, lower, upper,
                        color='gray', alpha=0.3, label='95% 置信区间'
                    )
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
                    ax_loss.plot(history.history['val_loss'], label='验证损失', color='red', linewidth=2)
                    ax_loss.set_title('模型训练损失曲线', fontsize=14, fontweight='bold')
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
                    
                    # 下载训练集图表
                    with col1:
                        buf_train = BytesIO()
                        fig_train.savefig(buf_train, format='pdf', dpi=300, bbox_inches='tight')
                        buf_train.seek(0)
                        st.download_button(
                            label="📥 下载训练集图表 (PDF)",
                            data=buf_train,
                            file_name="lstm_train_results.pdf",
                            mime="application/pdf"
                        )
                    
                    # 下载测试集图表
                    with col2:
                        buf_test = BytesIO()
                        fig_test.savefig(buf_test, format='pdf', dpi=300, bbox_inches='tight')
                        buf_test.seek(0)
                        st.download_button(
                            label="📥 下载测试集图表 (PDF)",
                            data=buf_test,
                            file_name="lstm_test_results.pdf",
                            mime="application/pdf"
                        )
                    
                    # 下载预测结果CSV
                    with col3:
                        results_df = pd.DataFrame({
                            'Date': preprocessor.test_dates,
                            'Actual': y_test_actual,
                            'Predicted': y_test_pred,
                            'Lower_95CI': lower,
                            'Upper_95CI': upper,
                            'Error': y_test_actual - y_test_pred
                        })
                        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 下载预测结果 (CSV)",
                            data=csv,
                            file_name="lstm_predictions.csv",
                            mime="text/csv"
                        )
                    
                    plt.close('all')
                    
                except Exception as e:
                    st.error(f"训练过程中出现错误: {str(e)}")
                    st.exception(e)
                
    except Exception as e:
        st.error(f"数据加载错误: {str(e)}")

else:
    st.info("请上传Excel数据文件开始预测")

# 清理临时文件
if uploaded_file is not None and 'tmp_path' in locals():
    try:
        os.unlink(tmp_path)
    except:
        pass