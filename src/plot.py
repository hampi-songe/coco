# 导入库函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('ggplot')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

seed=4
# 'SimSun' 是宋体的英文名，'Times New Roman' 是英文/数字的字体名
font_zh = {'family': 'SimSun', 'size': 17, 'weight': 'bold'}

# 同时建议设置全局字体为 Times New Roman，这样坐标轴的数字和图例会自动使用它
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

# --- 绘图部分 ---
# 设置画布大小时可以指定初始分辨率
fig = plt.figure(figsize=(8, 6), dpi=100) 

# 绘制曲线 (与之前一致)
plt.plot(df10['Step'], v_mean, label='MSC', color='#FF0000', linewidth=3)

# --- 图例设置 (高透明度 + 加粗) ---
legend_font = {
    'family': 'Times New Roman',
    'size': 14,
    'weight': "bold",
}
plt.legend(
    prop=legend_font, 
    framealpha=0.2,    # 透明度提高 (值越小越透明)
    facecolor='white', 
    edgecolor='none',  # 去掉图例边框线使画面更干净
    loc='lower right'  # 根据数据位置可调整
)

plt.grid(True, linestyle='--', alpha=0.5)

# --- 设置 X, Y 轴标签 (应用宋体加粗) ---
plt.xlabel("时间步个数", fontdict=font_zh)
plt.ylabel("平均测试胜率", fontdict=font_zh)

# --- 设置坐标轴刻度字体 (Times New Roman + 加粗) ---
plt.xticks([0, 2e5, 4e5, 6e5, 8e5, 1e6], ["0", "0.2M", "0.4M", "0.6M", "0.8M", "1M"],
           fontsize=17, fontname='Times New Roman', fontweight='bold')
plt.yticks(fontsize=17, fontname='Times New Roman', fontweight='bold')

# 限制范围
plt.xlim(0, 1050000)
plt.ylim(0, 1)

# --- 保存图片 (高分辨率设置) ---
# dpi=600 是高清出版标准
plt.savefig(r'D:\desktop\_xq_1c3s5z.pdf', bbox_inches='tight', dpi=600)
plt.savefig(r'D:\desktop\_xq_1c3s5z.png', bbox_inches='tight', dpi=600) # 同时存一个高规格PNG备用
plt.show()