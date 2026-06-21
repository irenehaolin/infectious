"""
疾病发病率预测模型 - Streamlit多页应用
========================================
主应用入口文件
"""

import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="疾病发病率预测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体支持
st.markdown("""
<style>
    .main {
        font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
    }
    .stMarkdown {
        font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# 主页内容
st.title("🏥 疾病发病率预测系统")
st.markdown("---")

st.markdown("""
### 系统简介

本系统提供四种先进的疾病发病率预测模型，支持时间序列预测和可视化分析：

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **LSTM** | 长短期记忆网络，擅长捕捉长期依赖关系 | 单变量时间序列预测 |
| **SARIMAX** | 季节性自回归移动平均模型，支持外生变量 | 具有季节性特征的预测 |
| **SARIMAX-ATTLSTM** | 混合模型，结合统计模型和深度学习优势 | 复杂多变量预测场景 |
| **Transformer** | 基于注意力机制，全局信息建模能力强 | 多变量复杂预测任务 |

### 使用步骤

1. **查看使用说明** - 了解数据格式要求和各模型适用对象
2. **上传数据文件** - 准备符合格式要求的Excel数据文件
3. **选择模型** - 根据预测需求选择合适的模型
4. **配置参数** - 设置模型训练参数
5. **运行预测** - 执行模型训练和预测
6. **下载结果** - 保存预测结果和可视化图表

### 导航菜单

请在左侧边栏选择相应的功能页面：

- 📖 **模型使用说明** - 详细的数据格式和操作指南
- 🔮 **LSTM模型** - LSTM预测模型
- 📊 **SARIMAX模型** - SARIMAX预测模型
- 🎯 **SARIMAX-ATTLSTM模型** - 混合预测模型
- 🤖 **Transformer模型** - Transformer预测模型
""")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    © 2024 疾病发病率预测系统 | 基于Streamlit构建
</div>
""", unsafe_allow_html=True)