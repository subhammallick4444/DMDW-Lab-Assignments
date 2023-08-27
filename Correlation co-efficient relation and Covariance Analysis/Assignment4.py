import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
plt.style.use('seaborn-whitegrid')

df=pd.read_csv("Downloads/student-mat.csv")

df.head()

df.info()

dfn=df[['traveltime','studytime']]

dfn.head()

x=dfn['traveltime']
y=dfn['studytime']
sbs.lineplot(dfn)
plt.show()

from scipy.stats import norm
sbs.lineplot(x=dfn['traveltime'],y=dfn['studytime'],dashes=True)

corelation=dfn.corr()
print(corelation)

sbs.heatmap(corelation,cmap ="YlGnBu")
plt.show()

covar=dfn.cov()
print(covar)

sbs.heatmap(covar,cmap='YlGnBu')
plt.show()
