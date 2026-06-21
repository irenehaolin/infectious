"""
SARIMAX模型预测页面
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
    page_title="SARIMAX模型预测",
    page_icon="📊",
    layout="wide"
)

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.title("📊 SARIMAX模型预测")
st.markdown("---")

# 侧边栏参数设置
st.sidebar.header("模型参数设置")

# 参数设置
st.sidebar.subheader("非季节性参数")
p = st.sidebar.slider("p (AR阶数)", min_value=0, max_value=3, value=2)
d = st.sidebar.slider("d (差分阶数)", min_value=0, max_value=2, value=0)
q = st.sidebar.slider("q (MA阶数)", min_value=0, max_value=3, value=2)

st.sidebar.subheader("季节性参数")
P = st.sidebar.slider("P (季节性AR阶数)", min_value=0, max_value=3, value=2)
D = st.sidebar.slider("D (季节性差分阶数)", min_value=0, max_value=2, value=0)
Q = st.sidebar.slider("Q (季节性MA阶数)", min_value=0, max_value=3, value=1)
s = st.sidebar.slider("s (季节周期)", min_value=4, max_value=52, value=52)

st.sidebar.subheader("其他参数")
test_size = st.sidebar.slider("测试集大小", min_value=10, max_value=52, value=26)
use_grid_search = st.sidebar.checkbox("使用网格搜索自动选参", value=False)

# 网格搜索参数范围
if use_grid_search:
    st.sidebar.subheader("网格搜索范围")
    p_range = st.sidebar.multiselect("p范围", [0, 1, 2, 3], default=[0, 1, 2])
    d_range = st.sidebar.multiselect("d范围", [0, 1, 2], default=[0, 1])
    q_range = st.sidebar.multiselect("q范围", [0, 1, 2, 3], default=[0, 1, 2])
    P_range = st.sidebar.multiselect("P范围", [0, 1, 2, 3], default=[0, 1, 2])
    D_range = st.sidebar.multiselect("D范围", [0, 1, 2], default=[0, 1])
    Q_range = st.sidebar.multiselect("Q范围", [0, 1, 2, 3], default=[0, 1, 2])

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
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    from sklearn.preprocessing import MinMaxScaler
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    from itertools import product
                    
                    status_text.text("正在加载数据...")
                    progress_bar.progress(10)
                    
                    # 数据处理
                    data_processed = data[['Date', 'Influenza']].copy()
                    data_processed['Date'] = pd.to_datetime(data_processed['Date'])
                    data_processed.set_index('Date', inplace=True)
                    
                    # log变换
                    data_processed['log_Influenza'] = np.log1p(data_processed['Influenza'])
                    
                    # 拆分训练和测试
                    train_data = data_processed.iloc[:-test_size].copy()
                    test_data = data_processed.iloc[-test_size:].copy()
                    
                    # 归一化
                    scaler = MinMaxScaler()
                    train_data['log_Influenza_scaled'] = scaler.fit_transform(train_data[['log_Influenza']])
                    test_data['log_Influenza_scaled'] = scaler.transform(test_data[['log_Influenza']])
                    data_processed['log_Influenza_scaled'] = scaler.transform(data_processed[['log_Influenza']])
                    
                    progress_bar.progress(20)
                    
                    if use_grid_search:
                        status_text.text("正在进行网格搜索...")
                        
                        # 网格搜索函数
                        def sarimax_grid_search(data, p_range, d_range, q_range, P_range, D_range, Q_range, s):
                            orders = list(product(p_range, d_range, q_range))
                            seasonal_orders = list(product(P_range, D_range, Q_range, [s]))
                            
                            results = []
                            best_aic = np.inf
                            best_order = None
                            best_seasonal_order = None
                            best_model = None
                            
                            total_combinations = len(orders) * len(seasonal_orders)
                            
                            for i, order in enumerate(orders):
                                for seasonal_order in seasonal_orders:
                                    try:
                                        model = SARIMAX(data, 
                                                       order=order, 
                                                       seasonal_order=seasonal_order,
                                                       enforce_stationarity=False,
                                                       enforce_invertibility=False)
                                        results_model = model.fit(disp=False, maxiter=200)
                                        
                                        results.append({
                                            'order': order,
                                            'seasonal_order': seasonal_order,
                                            'AIC': results_model.aic
                                        })
                                        
                                        if results_model.aic < best_aic:
                                            best_aic = results_model.aic
                                            best_order = order
                                            best_seasonal_order = seasonal_order
                                            best_model = results_model
                                            
                                    except:
                                        pass
                                
                                progress = 20 + int((i / len(orders)) * 40)
                                progress_bar.progress(progress)
                                status_text.text(f"网格搜索进度: {i+1}/{len(orders)}")
                            
                            return best_order, best_seasonal_order, best_model
                        
                        best_order, best_seasonal_order, sarimax_fit = sarimax_grid_search(
                            train_data['log_Influenza_scaled'],
                            p_range, d_range, q_range,
                            P_range, D_range, Q_range, s
                        )
                        
                        st.info(f"最优参数: order={best_order}, seasonal_order={best_seasonal_order}")
                        
                    else:
                        status_text.text("正在构建SARIMAX模型...")
                        progress_bar.progress(30)
                        
                        # 使用指定参数构建模型
                        order = (p, d, q)
                        seasonal_order = (P, D, Q, s)
                        
                        sarimax_model = SARIMAX(
                            train_data['log_Influenza_scaled'],
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        
                        status_text.text("正在拟合模型...")
                        progress_bar.progress(50)
                        
                        sarimax_fit = sarimax_model.fit(disp=False)
                        
                        best_order = order
                        best_seasonal_order = seasonal_order
                    
                    progress_bar.progress(70)
                    status_text.text("正在进行预测...")
                    
                    # 训练集预测
                    sarimax_train_pred = sarimax_fit.predict(start=0, end=len(train_data)-1)
                    
                    # 测试集预测
                    sarimax_forecast = sarimax_fit.get_forecast(steps=len(test_data))
                    sarimax_test_pred = sarimax_forecast.predicted_mean
                    test_conf_int = sarimax_forecast.conf_int(alpha=0.05)
                    
                    progress_bar.progress(80)
                    
                    # 反归一化 + 反对数
                    train_pred_log = scaler.inverse_transform(sarimax_train_pred.values.reshape(-1, 1))
                    train_pred = np.expm1(train_pred_log)
                    
                    test_pred_log = scaler.inverse_transform(sarimax_test_pred.values.reshape(-1, 1))
                    test_pred = np.expm1(test_pred_log)
                    
                    # 反归一化预测区间
                    test_conf_int_log_lower = scaler.inverse_transform(test_conf_int.iloc[:, 0].values.reshape(-1, 1))
                    test_conf_int_log_upper = scaler.inverse_transform(test_conf_int.iloc[:, 1].values.reshape(-1, 1))
                    test_conf_int_lower = np.expm1(test_conf_int_log_lower)
                    test_conf_int_upper = np.expm1(test_conf_int_log_upper)
                    
                    # 实际值
                    train_actual = np.expm1(scaler.inverse_transform(train_data['log_Influenza_scaled'].values.reshape(-1, 1)))
                    test_actual = np.expm1(scaler.inverse_transform(test_data['log_Influenza_scaled'].values.reshape(-1, 1)))
                    
                    progress_bar.progress(90)
                    
                    # 计算评估指标
                    mae_train = mean_absolute_error(train_actual, train_pred)
                    rmse_train = np.sqrt(mean_squared_error(train_actual, train_pred))
                    r2_train = r2_score(train_actual, train_pred)
                    
                    mae_test = mean_absolute_error(test_actual, test_pred)
                    rmse_test = np.sqrt(mean_squared_error(test_actual, test_pred))
                    r2_test = r2_score(test_actual, test_pred)
                    
                    progress_bar.progress(100)
                    status_text.text("训练完成！")
                    
                    # 显示模型摘要
                    st.markdown("---")
                    st.header("模型信息")
                    
                    st.write(f"**模型参数**: SARIMAX{best_order}x{best_seasonal_order}")
                    st.write(f"**AIC**: {sarimax_fit.aic:.2f}")
                    st.write(f"**BIC**: {sarimax_fit.bic:.2f}")
                    
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
                    
                    fig_train, ax_train = plt.subplots(figsize=(14, 6))
                    ax_train.plot(train_data.index, train_actual.flatten(), label='实际值', color='blue', linewidth=2)
                    ax_train.plot(train_data.index, train_pred.flatten(), label='拟合值', color='red', linewidth=2, alpha=0.8)
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
                    ax_test.plot(test_data.index, test_actual.flatten(), label='实际值', color='blue', linewidth=2)
                    ax_test.plot(test_data.index, test_pred.flatten(), label='预测值', color='orange', linewidth=2)
                    ax_test.fill_between(test_data.index,
                                         test_conf_int_lower.flatten(),
                                         test_conf_int_upper.flatten(),
                                         color='gray', alpha=0.3, label='95% 预测区间')
                    ax_test.set_title('Influenza - 测试集预测效果（含95%置信区间）', fontsize=14, fontweight='bold')
                    ax_test.set_xlabel('时间', fontsize=12)
                    ax_test.set_ylabel('发病率', fontsize=12)
                    ax_test.legend(fontsize=12)
                    ax_test.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig_test)
                    
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
                            file_name="sarimax_train_results.pdf",
                            mime="application/pdf"
                        )
                    
                    with col2:
                        buf_test = BytesIO()
                        fig_test.savefig(buf_test, format='pdf', dpi=300, bbox_inches='tight')
                        buf_test.seek(0)
                        st.download_button(
                            label="📥 下载测试集图表 (PDF)",
                            data=buf_test,
                            file_name="sarimax_test_results.pdf",
                            mime="application/pdf"
                        )
                    
                    with col3:
                        results_df = pd.DataFrame({
                            'Date': test_data.index,
                            'Actual': test_actual.flatten(),
                            'Predicted': test_pred.flatten(),
                            'Lower_95CI': test_conf_int_lower.flatten(),
                            'Upper_95CI': test_conf_int_upper.flatten(),
                            'Error': test_actual.flatten() - test_pred.flatten()
                        })
                        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 下载预测结果 (CSV)",
                            data=csv,
                            file_name="sarimax_predictions.csv",
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