import pandas as pd
dataset=pd.read_csv("Downloads/iris.csv")
dataset.head(150)

n_num=[3.5,3.0,3.2,3.1,3.6]
n=len(n_num)
get_sum=sum(n_num)
mean = get_sum / n
print("mean / Average is: " + str(mean))

n_num=[3.5,3.0,3.2,3.1,3.6]
n = len(n_num)
n_num.sort()
if n % 2 == 0:
    median1 = n_num[n//2]
    median2 = n_num[n//2 - 1]
    median = (median1 + median2)/2
else:
    median = n_num[n//2]
    print("Median is: " + str(median))

from collections import Counter
n_num=[3.5,3.0,3.2,3.1,3.6]
n=len(n_num)
data=Counter(n_num)
get_mode=dict(data)
mode=[k for k, v in get_mode.items() if v==max(list(data.values()))]
if len(mode)==n:
    get_mode="No mode found"
else:
    get_mode="Mode is / are: " + ', '.join(map(str, mode))
print(get_mode)

import pandas as pd
import matplotlib.pyplot as plot
dataset=pd.read_csv("Downloads/iris.csv")
b_plot=dataset.boxplot(column=['sepal_length','sepal_width','petal_length','petal_width'])
b_plot.plot()
plot.show()

