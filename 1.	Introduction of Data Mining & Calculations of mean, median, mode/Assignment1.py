#mean without using library
n_num = [1,2,3,4,5]
n = len(n_num)
get_sum = sum(n_num)
mean = get_sum / n
print("Mean / Average is: " + str(mean))

#median without using library
n_num = [1,2,3,4,5]
n = len(n_num)
n_num.sort()
if n % 2 == 0:
    median1 = n_num[n//2]
    median2 = n_num[n//2 - 1]
    median = (median1 + median2)/2
else:
    median = n_num[n//2]
    print("Median is: " + str(median))

#python program to print mode of elements
from collections import Counter
n_num=[1,2,3,4,5,5]
n=len(n_num)
data=Counter(n_num)
get_mode=dict(data)
mode=[k for k, v in get_mode.items() if v==max(list(data.values()))]
if len(mode)==n:
    get_mode="No mode found"
else:
    get_mode="Mode is / are: " + ', '.join(map(str, mode))
print(get_mode)

#mode with library function 
import numpy
speed=[99,86,87,88,111,86,103,87,94,78,77,85,86]
x=numpy.mean(speed)
y=numpy.median(speed)
s=numpy.std(speed)
v=numpy.var(speed)
print(x)
print(y)
print(s)
print(v)

#mode with library function
from scipy import stats
speed=[99,86,87,88,111,86,103,87,94,78,77,85,86]
x = stats.mode(speed)
print(x)
