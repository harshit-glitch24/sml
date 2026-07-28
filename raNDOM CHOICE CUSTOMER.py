#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data = {
    'Satisfaction Level': np.random.choice(
        ['Low', 'Medium', 'High'],
        size=200
    ),
    'Repeat Purchase': np.random.choice(
        ['Yes', 'No'],
        size=200,
        p=[0.4, 0.6]  
    )
}


# In[3]:


df_customer = pd.DataFrame(data)


cross_tab = pd.crosstab(
    df_customer['Satisfaction Level'],
    df_customer['Repeat Purchase']
)


# In[4]:


cross_tab.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Repeat Purchase by Satisfaction Level')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Repeat Purchase')
plt.tight_layout()
plt.show()


# In[5]:


cross_tab.plot(kind='bar', stacked=False, figsize=(8, 6))
plt.title('Repeat Purchase by Satisfaction Level')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Repeat Purchase')
plt.tight_layout()
plt.show()


# In[6]:


print("""
Inference:
The bar charts show the distribution of repeat purchases across different customer satisfaction levels.
Since the dataset is randomly generated, there is no strong relationship between customer satisfaction and repeat purchases.
The stacked bar chart shows the total number of customers in each satisfaction category along with the proportion of repeat and non-repeat purchases.
The grouped bar chart makes it easier to compare the number of customers who made repeat purchases ('Yes') and those who did not ('No') for each satisfaction level.
Overall, the charts help visualize customer behaviour, but no meaningful conclusion can be drawn because the data is randomly generated.
""")


# In[ ]:




