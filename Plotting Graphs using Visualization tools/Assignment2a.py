import pandas as pd 
d = pd.read_csv("DownloadS/Toyota.csv")
d

d.head(100)

import matplotlib.pyplot as plt
plt.scatter(d['Age'],d['Price'],c='blue')
plt.title('Scatter plot of Price vs Age of the Cars')
plt.xlabel('Age (months)')
plt.ylabel('Price (Euros)')
plt.show()

plt.hist(d['KM'], color = 'blue', edgecolor = 'white', bins = 5)
plt.title('Histogram of Kilometer')
plt.xlabel('Kilometer')
plt.ylabel('Frequency')
plt.show()

import numpy as np
counts = [979,120,12]
fuelType = ("Petrol","Diesel","CNG")
index = np.arange(len(fuelType))
plt.bar(index, counts, color=['black', 'blue', 'cyan'])
plt.title("Bar plot of fuel types")
plt.xlabel("Fuel Types")
plt.ylabel("Frequency")
plt.xticks(index,fuelType,rotation = 90)
plt.show()

# Bar plot generation using countplot function
sns.countplot(x="FuelType",data=d)

sns.boxplot(y=d["Price"])

sns.countplot(x="FuelType", data=d, hue="Automatic")

sns.boxplot(x =d['FuelType'], y =d["Price"])

sns.boxplot(x = "FuelType", y =d["Price"], hue = "Automatic", data =d)

