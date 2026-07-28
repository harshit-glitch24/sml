#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df_customer = pd.read_csv("Customer Satisfaction Scores and Behavior Data.csv")

# Cross-tabulation
cross_tab = pd.crosstab(
    df_customer['Loyalty_Level'],
    df_customer['Purchase_History']
)

print("Cross Tabulation:")
print(cross_tab)

# Stacked Bar Chart
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6))
plt.title('Purchase History by Loyalty Level')
plt.xlabel('Loyalty Level')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Purchase History')
plt.tight_layout()
plt.show()

# Grouped Bar Chart
cross_tab.plot(kind='bar', stacked=False, figsize=(8,6))
plt.title('Purchase History by Loyalty Level')
plt.xlabel('Loyalty Level')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Purchase History')
plt.tight_layout()
plt.show()


# In[2]:


print("""
Inference:
The charts show the relationship between customer loyalty level and purchase history.
Customers with High loyalty generally tend to have a stronger purchase history compared to customers with Low loyalty.
Medium loyalty customers exhibit a moderate purchase pattern.
The stacked bar chart displays the overall distribution of purchase history within each loyalty level, while the grouped bar chart allows easy comparison between purchase history categories.
Overall, customer loyalty appears to influence purchase behaviour, with higher loyalty levels generally associated with better purchase history.
""")


# In[ ]:




