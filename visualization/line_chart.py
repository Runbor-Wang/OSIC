from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

x_axis_data = [0, 1, 2, 3, 4, 5, 6]
y_axis_data_0 = [122.2, 123.1, 123.7, 123.1, 123.9, 123.3, 123.3]
y_axis_data_1 = [122.2, 122.6, 123.5, 123.7, 122.8, 123.8, 122.2]
y_axis_data_2 = [122.2, 123.1, 123.1, 122.4, 122.5, 121.2, 119.8]
y_axis_data_3 = [122.2, 124.2, 123.9, 123.9, 122.7, 122.9, 122.5]

plt.plot(x_axis_data, y_axis_data_0, 'ro-', color='royalblue', alpha=0.8, label='Spatial')
plt.plot(x_axis_data, y_axis_data_1, 'ro-', color='forestgreen', alpha=0.8, label='Spatial')
plt.plot(x_axis_data, y_axis_data_2, 'ro-', color='darkorange', alpha=0.8, label='Spatial')
plt.plot(x_axis_data, y_axis_data_3, 'ro-', color='indianred', alpha=0.8, label='Spatial')

for x, y in zip(x_axis_data, y_axis_data_0):
    plt.text(x, y+0.1, '%.1f' % y, color='royalblue', ha='center', va='bottom', fontsize=7.5)
for x, y in zip(x_axis_data, y_axis_data_1):
    plt.text(x, y+0.1, '%.1f' % y, color='forestgreen', ha='center', va='bottom', fontsize=7.5)
for x, y in zip(x_axis_data, y_axis_data_2):
    plt.text(x, y+0.1, '%.1f' % y, color='darkorange', ha='center', va='bottom', fontsize=7.5)
for x, y in zip(x_axis_data, y_axis_data_3):
    plt.text(x, y+0.1, '%.1f' % y, color='indianred', ha='center', va='bottom', fontsize=7.5)

plt.legend(["spatial", "channel", "parallel", "cascade"])
plt.xlabel('layers')
plt.ylabel('CIDEr')

plt.show()
plt.savefig('line_chart.jpg')  # 保存该图片
plt.savefig('line_chart.pdf', bbox_inches='tight')  # 保存该图片
plt.close()
