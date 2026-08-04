#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a fixed dataset
data = {
    'Engine Size (L)': [1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0],
    'Fuel Efficiency (MPG)': [42, 40, 38, 35, 32, 30, 27, 25, 22, 20],
    'Car Price': [12000, 15000, 18000, 22000, 25000, 28000, 32000, 35000, 42000, 50000]
}


# In[2]:


df_cars = pd.DataFrame(data)


# In[3]:


print(df_cars)


# In[4]:


correlation_matrix = df_cars.corr()


# In[5]:


print("\nCorrelation Matrix")
print(correlation_matrix)


# In[6]:


plt.figure(figsize=(6,5))


# In[7]:


sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f')


# In[8]:


sns.pairplot(df_cars,
             diag_kind='hist')

plt.show()


# In[9]:


print("""
Final Inference:

• Engine Size and Car Price have a strong positive relationship, indicating that cars with larger engines are generally more expensive.

• Engine Size and Fuel Efficiency have a strong negative relationship, showing that larger engines consume more fuel and therefore have lower mileage.

• Fuel Efficiency and Car Price also have a strong negative relationship, suggesting that higher-priced cars in this dataset tend to have lower fuel efficiency.

• The heatmap provides the numerical strength and direction of these relationships, while the pairplot visually confirms the same relationships through scatter plots and shows the distribution of each variable on the diagonal.

• Together, the heatmap and pairplot help identify trends, correlations, and patterns useful for Exploratory Data Analysis (EDA).
""")


# In[ ]:




