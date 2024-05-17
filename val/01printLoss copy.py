import matplotlib.pyplot as plt

# 区间和数据点
# intervals = ['0-0.05', '0.05-0.21', '0.21-0.33', '0.33-0.61', '0.61-0.79', '0.79-1']

# intervals = ['0-0.07', '0.07-0.29', '0.29-0.36', '0.36-0.68', '0.68-0.79', '0.79-1']

intervals = ['0-0.12', '0.12-0.36', '0.36-0.44', '0.44-0.57', '0.57-0.64', '0.64-1']
# percentages = [0.11, 0.21, 0.46, 0.13,0.07,0.02]
percentages = [0.12, 0.22, 0.31, 0.08,0.04,0.03]
# percentages = [0.12, 0.24, 0.32, 0.16,0.09,0.07]
# percentages = [0.32, 0.26, 0.18, 0.12,0.11,0.01]
# 绘制柱状图
plt.bar(range(len(percentages)), percentages, color='grey', alpha=0.5)

# 绘制折线图
plt.plot(range(len(percentages)), percentages, marker='o', color='orange', linestyle='-', linewidth=2)

# 添加百分比标签
for i, value in enumerate(percentages):
    plt.text(i, value + 0.03, f'{value:.2f}%', ha='center')

# 添加标题和标签
plt.xlabel('Intervals', fontsize=16)
plt.ylabel('Percentage of Data Points', fontsize=16)
# plt.title('Distribution of Data Points in Intervals')
plt.xticks(range(len(intervals)), intervals, rotation=45, fontsize=16)
plt.yticks(fontsize=16)
# plt.legend()

# 设置y轴刻度为0~1
plt.ylim(0, 0.6)

# 显示图形
plt.tight_layout()
plt.savefig('dist-basestation.pdf', dpi=1200, bbox_inches='tight')
