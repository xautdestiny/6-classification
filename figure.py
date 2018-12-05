import numpy as np
import matplotlib.pyplot as plt

x = range(1,11)  # 横轴的数据
for i in x:
    y = i * i

    plt.plot(i, y,'*')  # 调用pylab的plot函数绘制曲线
plt.show()