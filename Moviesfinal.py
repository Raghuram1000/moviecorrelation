#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd  # Correct alias for pandas
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt# Correct import for pyplot
import numpy as np
# Set plot style
plt.style.use('ggplot')

# Set figure size
matplotlib.rcParams['figure.figsize'] = (12, 8)

# Ensure you're in a Jupyter Notebook to use this command
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df=pd.read_csv(r'C:\Users\AVLN RAGHURAM\Downloads\archive (1)\movies.csv')


# In[11]:


df.head()


# In[20]:


for col in df.columns:
    pctmissing = np.mean(df[col].isnull())
    print('{}-{}%'.format(col, pctmissing))


# In[22]:


df.dtypes


# In[24]:


df['budget'] = df['budget'].fillna(0).astype('int64')
df['gross'] = df['gross'].fillna(0).astype('int64')


# In[25]:


df


# In[30]:


df.sort_values(by=['gross'],inplace=False,ascending=False)


# In[33]:


pd.set_option('display.max_rows',None)


# In[35]:


df['company'].drop_duplicates().sort_values(ascending=False)


# In[37]:


plt.scatter(x=df['budget'],y=df['gross'])

plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for film')
plt.show()


# In[41]:


sns.regplot(x=df['budget'],y=df['gross'],data=df,scatter_kws={"color":"red"},line_kws={"color":"blue"})


# In[44]:


df_numeric = df.select_dtypes(include=['number'])
df_numeric.corr()


# In[53]:


correlationmatrix=df_numeric.corr()
sns.heatmap(correlationmatrix,annot=True)
plt.title('Correlation matrix for numeric features')
plt.xlabel('movie features')
plt.ylabel('movie features')
plt.show()


# In[51]:


dfnumerized = df.copy()  # Create a copy to avoid modifying the original DataFrame

for colname in dfnumerized.columns:
    if dfnumerized[colname].dtype == 'object':  # Check for object type columns
        print(f'Converting {colname} to category')
        dfnumerized[colname] = dfnumerized[colname].astype('category')  # Convert to category
        dfnumerized[colname] = dfnumerized[colname].cat.codes  # Replace with category codes


# In[54]:


dfnumerized 


# In[55]:


correlationmatrix=dfnumerized.corr()
sns.heatmap(correlationmatrix,annot=True)
plt.title('Correlation matrix for numeric features')
plt.xlabel('movie features')
plt.ylabel('movie features')
plt.show()


# In[56]:


correlationmat=dfnumerized.corr()
corrpairs=correlationmat.unstack()
corrpairs


# In[57]:


highcorr=corrpairs[(corrpairs)>0.5]
highcorr

