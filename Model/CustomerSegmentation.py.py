#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


dset=pd.read_csv('Mall_Customers.csv')


# In[3]:


dset.head()


# In[4]:


dset['CustomerID'].isnull().sum()


# In[5]:


dset.isna().sum()


# # The Dataset Does not contain any null values

# In[6]:


dset.info()


# In[7]:


dset.describe()


# # CustomerId Age And Gender Does not affect the Segementation process. The Segementation only depends on Annual Income and Spending Score

# In[8]:


x=dset.iloc[:,[3,4]].values


# In[9]:


x


# #Choosing number of clusters using WCSS parameter-With in Clusters sum of squares

# In[10]:


dset=dset[~dset['Spending Score (1-100)'].isnull()]


# In[11]:


dset


# In[12]:


#finding WCSS values for different number of clusters
wcss=[]
for i in range(1,11):
    kmeans=KMeans(i, init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[13]:


np.argmax(wcss)


# In[14]:


wcss


# #plot elbow graph which cluster has min values

# In[15]:


sns.set()
plt.plot(range(1,11),wcss)
plt.title('The Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# optimal number of clusters=5 beacuse it is one of the elbow point

# In[16]:


Kmeans1=KMeans(5,init='k-means++',random_state=7)


# In[17]:


Kmeans1.fit(x)


# In[18]:


y=Kmeans1.fit_predict(x)


# In[19]:


y


# visualizing all the cluster

# In[20]:


plt.figure(figsize=(8,8))
plt.scatter(x[y==0,0],x[y==0,1],s=50,c='green',label='Cluster-1')
plt.scatter(x[y==1,0],x[y==1,1],s=50,c='yellow',label='Cluster-2')
plt.scatter(x[y==2,0],x[y==2,1],s=50,c='blue',label='Cluster-3')
plt.scatter(x[y==3,0],x[y==3,1],s=50,c='pink',label='Cluster-4')
plt.scatter(x[y==4,0],x[y==4,1],s=50,c='black',label='Cluster-5')

plt.scatter(Kmeans1.cluster_centers_[:,0],Kmeans1.cluster_centers_[:,1],s=100,c='cyan',label='Centroid')
#represents x-axis and y-axis values of cetnroids
plt.title('Customer-Segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Customer Score')
plt.show()


# # Here x[y=0,0] represents elements having cluster label as 0 and and second 0 represents the coloumn number in x similarly with 0,1 and so on

# In[ ]:




