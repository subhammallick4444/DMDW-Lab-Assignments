#for import the files and read the files
import pandas as pd 
ds=pd.read_csv("Downloads/student-mat.csv")
ds.head(400)

import pandas as pd
import numpy as np
import seaborn as sbs
import matplotlib.pyplot as plt
ds1=pd.read_csv("Downloads/student-mat.csv",sep=',')
ds1.head()

#for binary attributes
ds=ds1[['schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']]
ds.head()
ds.info()

ds=ds.replace('no', 0)
ds=ds.replace('yes',1)
ds.head()
ds.info

n=np.array(ds[['schoolsup','famsup']])
n=n.reshape(-1,2)
n.shape

m=np.array(ds[['romantic','internet']])
m=m.reshape(-1,2)
m.shape

from scipy.spatial import distance
dist_matrix = distance.cdist(n,m)
from scipy.spatial import distance
dist_matrix = distance.cdist(n,m)
dist_matrix.shapesbs.heatmap(dist_matrix)
plt.show()

sbs.heatmap(dist_matrix)
sns.color_palette("blend:#7AB,#EDA", as_cmap=True)
plt.show()


#for numeric data
numeric=ds1[['age','Medu','Fedu','traveltime','studytime','failures']]
numeric.head()

numeric.info()

num1=np.array(numeric[['age','failures']])
num1.reshape(-1,2)
num1.shape

num2=np.array(numeric[['Fedu','Medu']])
num2.reshape(-1,2)
num2.shape

from scipy.spatial import distance
dist_matrix = distance.cdist(num1,num2)
dist_matrix.shape
sbs.heatmap(dist_matrix)

#for nominal attribute
nominal=ds1[['Mjob','Fjob','reason','guardian']]
nominal=nominal.replace('at_home','home')
nominal=(nominal.astype('category'))


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
nominal['guardian']=lb.fit_transform(nominal['guardian'])


nominal['Mjob']=lb.fit_transform(nominal['Mjob'])
nominal['Fjob']=lb.fit_transform(nominal['Fjob'])
nominal['reason']=lb.fit_transform(nominal['reason'])

nominal.head()


nom1=np.array(nominal)
nom1.reshape(-1,2)


nom2=np.array(nominal)
nom2.reshape(-1,2)

from scipy.spatial import distance
dist_matrix = distance.cdist(nom1,nom2)
sbs.heatmap(dist_matrix)
dist_matrix.shape
