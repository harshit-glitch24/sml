#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = {
    'Satisfaction Level': [
        'High', 'Medium', 'Low', 'High', 'Medium',
        'Low', 'High', 'Medium', 'Low', 'High',
        'Medium', 'Low', 'High', 'Medium', 'Low',
        'High', 'Medium', 'Low', 'High', 'Medium',
        'High', 'Low', 'Medium', 'High'
    ],

    'Repeat Purchase': [
        'Yes', 'No', 'No', 'Yes', 'Yes',
        'No', 'Yes', 'No', 'No', 'Yes',
        'Yes', 'No', 'Yes', 'No', 'No',
        'Yes', 'Yes', 'No', 'Yes', 'No',
        'No', 'Yes', 'Yes', 'Yes'
    ]
}

df_customer = pd.DataFrame(data)

df_customer


# In[3]:


cross_tab = pd.crosstab(
    df_customer['Satisfaction Level'],
    df_customer['Repeat Purchase']
)

print(cross_tab)


# In[4]:


cross_tab.plot(kind='bar', stacked=True, figsize=(8,6))

plt.title('Repeat Purchase by Satisfaction Level')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Repeat Purchase')
plt.tight_layout()

plt.show()


# In[5]:


cross_tab.plot(kind='bar', stacked=False, figsize=(8,6))

plt.title('Repeat Purchase by Satisfaction Level')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Repeat Purchase')
plt.tight_layout()

plt.show()


# In[10]:


print(""" INFERENCE
The analysis shows a relationship between customer satisfaction and repeat purchases.
Customers with High satisfaction have the highest number of repeat purchases ("Yes"),
indicating that satisfied customers are more likely to buy again.
Customers with Low satisfaction have more "No" responses than "Yes",
suggesting they are less likely to make repeat purchases.
Customers with Medium satisfaction show a mixed pattern, with both repeat and non-repeat purchases,
indicating moderate customer loyalty.
""")


# In[ ]:




