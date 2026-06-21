"""
模型使用说明页面
"""

import streamlit as st

st.set_page_config(
    page_title="模型使用说明",
    page_icon="📖",
    layout="wide"
)

st.title("📖 模型使用说明")
st.markdown("---")

# 数据格式说明
st.header("一、数据格式要求")

st.markdown("""
### 1. 数据文件格式

- **文件类型**: Excel文件 (.xlsx)
- **必需列**: 
  - `Date`: 日期列（格式：YYYY-MM-DD 或 YYYY/MM/DD）
  - `Influenza`: 目标变量（疾病发病率）

### 2. 可选特征列

以下特征列可用于多变量预测模型：

| 列名 | 说明 | 适用模型 |
|------|------|----------|
| `WAT` | 温度数据 | SARIMAX-ATTLSTM, Transformer |
| `WCR` | 降雨数据 | SARIMAX-ATTLSTM, Transformer |
| `X1` - `X7` | 辅助特征变量 | SARIMAX-ATTLSTM, Transformer |
| `Vaccine` | 疫苗接种数据 | SARIMAX-ATTLSTM, Transformer |

### 3. 数据示例

```
Date        | Influenza | WAT | WCR | X1 | X2 | ... | Vaccine
2020-01-01  | 1.23      | 15  | 0.5 | 10 | 20 | ... | 100
2020-01-08  | 1.45      | 14  | 0.3 | 12 | 22 | ... | 150
...
```

### 4. 数据要求

- 数据应为**周数据**（每周一个观测值）
- 建议至少有 **100+** 个观测值用于训练
- 数据应按时间顺序排列
- 缺失值需要提前处理或填充
""")

st.markdown("---")

# 模型适用对象
st.header("二、模型适用对象")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 LSTM模型")
    st.markdown("""
    **适用场景**:
    - 单变量时间序列预测
    - 数据具有长期依赖关系
    - 需要捕捉复杂的非线性模式
    
    **优点**:
    - 能够学习长期时间依赖
    - 对非线性模式敏感
    - Bootstrap置信区间估计
    
    **参数说明**:
    - `seq_len`: 序列长度（默认4）
    - `epochs`: 训练轮数（默认100）
    - `batch_size`: 批次大小（默认32）
    """)

    st.subheader("📊 SARIMAX模型")
    st.markdown("""
    **适用场景**:
    - 具有明显季节性特征的数据
    - 周数据（季节周期52周）
    - 需要统计推断和解释性
    
    **优点**:
    - 季节性模式捕捉能力强
    - 提供统计显著性检验
    - AIC/BIC准则自动选参
    
    **参数说明**:
    - `p,d,q`: ARIMA参数
    - `P,D,Q,s`: 季节性参数（s=52）
    """)

with col2:
    st.subheader("🎯 SARIMAX-ATTLSTM模型")
    st.markdown("""
    **适用场景**:
    - 多变量复杂预测任务
    - 需要结合统计模型和深度学习
    - 残差序列建模
    
    **优点**:
    - 混合模型优势互补
    - Attention机制增强特征选择
    - 多变量特征融合
    
    **参数说明**:
    - SARIMAX参数自动网格搜索
    - LSTM单元数：104
    - 序列长度：4
    """)

    st.subheader("🤖 Transformer模型")
    st.markdown("""
    **适用场景**:
    - 多变量复杂预测任务
    - 需要全局信息建模
    - 大规模数据预测
    
    **优点**:
    - 注意力机制全局建模
    - 多头注意力并行处理
    - 支持多特征输入
    
    **参数说明**:
    - `head_size`: 注意力头维度（默认32）
    - `num_heads`: 注意力头数量（默认4）
    - `num_layers`: Transformer层数（默认3）
    """)

st.markdown("---")

# 使用步骤
st.header("三、使用步骤")

st.markdown("""
### 步骤1: 数据准备

1. 准备符合格式要求的Excel数据文件
2. 确保数据按时间顺序排列
3. 检查并处理缺失值

### 步骤2: 选择模型

1. 在左侧导航栏选择对应的模型页面
2. 根据预测需求选择合适的模型

### 步骤3: 上传数据

1. 点击"上传数据文件"按钮
2. 选择准备好的Excel文件
3. 系统自动验证数据格式

### 步骤4: 配置参数

1. 根据需要调整模型参数
2. 设置训练参数（epochs、batch_size等）
3. 配置预测参数

### 步骤5: 运行预测

1. 点击"开始训练"按钮
2. 等待模型训练完成
3. 查看训练日志和进度

### 步骤6: 结果分析

1. 查看预测结果图表
2. 分析评估指标（MAE、RMSE、R²）
3. 下载预测结果和图表
""")

st.markdown("---")

# 评估指标说明
st.header("四、评估指标说明")

st.markdown("""
| 指标 | 说明 | 计算公式 |
|------|------|----------|
| **MAE** | 平均绝对误差 | $MAE = \\frac{1}{n}\\sum|y_i - \\hat{y}_i|$ |
| **RMSE** | 均方根误差 | $RMSE = \\sqrt{\\frac{1}{n}\\sum(y_i - \\hat{y}_i)^2}$ |
| **R²** | 决定系数 | $R^2 = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2}$ |
| **MAPE** | 平均绝对百分比误差 | $MAPE = \\frac{100}{n}\\sum\\frac{|y_i - \\hat{y}_i|}{y_i}$ |

**指标解读**:
- MAE和RMSE越小越好，表示预测误差小
- R²越接近1越好，表示模型拟合能力强
- MAPE越小越好，通常小于10%为优秀
""")

st.markdown("---")

# 注意事项
st.header("五、注意事项")

st.warning("""
**重要提示**:
1. 模型训练可能需要较长时间，请耐心等待
2. 深度学习模型（LSTM、Transformer）需要足够的训练数据
3. SARIMAX网格搜索会尝试多种参数组合，耗时较长
4. 建议先用小参数测试，确认流程后再调整参数
5. 下载的图表为PDF格式，预测结果为CSV格式
""")

st.info("""
**技术支持**:
- 如遇到问题，请检查数据格式是否符合要求
- 模型训练失败可能是参数设置不当或数据量不足
- 可尝试调整epochs、batch_size等参数
""")