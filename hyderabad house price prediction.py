#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data=pd.read_csv('HYDhousedata.csv')


# In[2]:


data.head()


# In[3]:


pd.set_option('display.max_columns',None)


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.isna().sum()


# In[7]:


data.columns


# In[8]:


data=data.drop(['active','amenities','combineDescription','deposit','facing','facingDesc','gym','id','isMaintenance','lift','property_age','reactivationSource','sharedAccomodation','shortUrl','swimmingPool','waterSupply'],axis=1)


# In[9]:


data.head(1)


# In[10]:


data=data.drop(['completeStreetName','loanAvailable','localityId','location','parking','parkingDesc','propertyTitle','weight'],axis=1)


# In[11]:


data.head()


# In[12]:


data=data.drop(['ownerName','propertyType'],axis=1)


# In[13]:


data.head()


# In[14]:


data['locality'].values.tolist()


# In[15]:


data.shape


# In[16]:


data.isna().sum()


# In[17]:


data=data.drop(['floor'],axis=1)


# In[18]:


data.head()


# In[19]:


data['maintenanceAmount'].isna().sum()


# In[20]:


data['location']=data['locality']


# In[21]:


data=data.drop(['locality'],axis=1)


# In[22]:


data['balconies'].unique()


# In[23]:


data=data.drop(['balconies'],axis=1)


# In[24]:


data.info()


# In[25]:


data.shape


# In[26]:


data.dropna(inplace=True)


# In[27]:


data.info()


# In[28]:


data=data.drop(['maintenanceAmount'],axis=1)


# In[29]:


data=data.drop(['totalFloor'],axis=1)


# In[30]:


data.info()


# In[31]:


data['property_size'].values.tolist()


# In[32]:


data['type_bhk'].values.tolist(
)


# In[33]:


data['type_bhk'].unique()


# In[34]:


data1=data


# In[35]:


v=pd.get_dummies(data['furnishingDesc'])
v=v.astype(int)
v.head()


# In[36]:


data2=pd.concat([data,v.drop(['Unfurnished'],axis=1)],axis='columns')


# In[37]:


data2.head()


# In[38]:


dummy=pd.get_dummies(data['type_bhk'])
dummy=dummy.astype(int)
dummy.head()


# In[39]:


data2=pd.concat([data2,dummy.drop(['RK1'],axis=1)],axis='columns')


# In[40]:


data2.head()


# In[41]:


data2=data2.drop(['furnishingDesc','type_bhk'],axis=1)


# In[42]:


data2.info()
data3=data2


# In[43]:


loc=pd.get_dummies(data2['location'])


loc=loc.astype(int)


# In[44]:


loc.head()


# In[45]:


data2=pd.concat([data2,loc.drop(['kondapur'],axis=1)],axis='columns')
data2.head()


# In[46]:


data2.info()


# In[47]:


data2=data2.drop(['location'],axis=1)


# In[48]:


X=data2.drop(['rent_amount'],axis=1)
X.head()


# In[49]:


y=data2.rent_amount


# In[50]:


y.head()


# In[51]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)


# In[74]:


from sklearn.ensemble import GradientBoostingRegressor

GBR=GradientBoostingRegressor()
GBR.fit(X_train,y_train)
GBR.score(X_test,y_test)


# In[75]:


X.columns


# In[76]:


np.where(X.columns=='BHK1')[0][0]


# In[77]:


def predict(location, bhk, furnishing, bathroom, propertysize):
    furnish_index = np.where(X.columns == furnishing)[0][0] if furnishing in X.columns else -1
    bhk_index = np.where(X.columns == bhk)[0][0] if bhk in X.columns else -1
    loc_index = np.where(X.columns == location)[0][0] if location in X.columns else -1

    x = np.zeros(len(X.columns))

    x[0] = bathroom
    x[1] = propertysize
    if furnish_index >= 0:
        x[furnish_index] = 1

    if bhk_index >= 0:
        x[bhk_index] = 1

    if loc_index >= 0:
        
        x[loc_index] = 1
 
    if furnishing == 'Unfurnished':
        x_default = np.zeros(len(X.columns))
        x_default[0] = bathroom
        x_default[1] = propertysize
        return model.predict([x_default])[0]

    if furnishing == 'RK':
        x_default = np.zeros(len(X.columns))
        x_default[0] = bathroom
        x_default[1] = propertysize
        return model.predict([x_default])[0]
    return model.predict([x])[0]


# In[84]:


predict('Uppal, ','BHK1','Full',3,1200)


# In[85]:


import pickle
pickle_out = open('predict.pkl',"wb")
pickle.dump(GBR,pickle_out)
pickle_out.close()




# In[80]:


X.columns.tolist()
pickle_out = open('predict.pkl',"wb")


# In[81]:


X.to_csv('kin.csv',index=False)


# In[82]:


len(X.columns)


# In[70]:





# In[ ]:




