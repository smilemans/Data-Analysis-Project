#!/usr/bin/env python
# coding: utf-8

# 分析背景：
# 电话服务公司、互联网服务提供商、付费电视公司、保险公司和警报监控服务通常使用客户流失分析和客户流失率作为其关键业务指标之一，因为保留现有客户的成本远低于获得新的一个客户。这些行业的公司经常设有客户服务部门，试图赢回叛逃客户，因为与新招募的客户相比，恢复的长期客户对公司具有更高的价值。
# 
# 分析目的：
# 通过建立流失模型，来预测客户流失，评估客户流失的风险。模型将生成了一个潜在流失客户的优先列表，它们可以有效地将客户保留营销计划集中在最易受客户流失影响的客户群中。
# 
# 数据来源：https://www.kaggle.com/blastchar/telco-customer-churn
# 
# 
# 

# # 1. 数据导入

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"]=["SimHei"]
matplotlib.rcParams["axes.unicode_minus"]=False


# In[2]:


telcom = pd.read_csv('C:/Users/24799/Desktop/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')  
telcom.head()


# In[3]:


#查看数据
print('数据量:\n {0}'.format(telcom.shape))
print('\n特征列表:\n {0}'.format(telcom.columns.tolist()))
print('\n缺失值数量:\n {0}'.format(telcom.isnull().sum().values.sum()))
print('\n唯一值数量:\n {0}'.format(telcom.nunique()))
print('\n每列数量:\n {0}'.format(telcom.count()))
print('\n特征类型:\n {0}'.format(telcom.dtypes))


# In[4]:


#删除TotalCharges列空格值所在的行
telcom = telcom.drop(telcom[telcom['TotalCharges']==' '].index.tolist())
telcom['TotalCharges'] = telcom['TotalCharges'].astype('float')
telcom['TotalCharges'].describe()


# # 2. 探索分析

# In[5]:


#将SeniorCitizen列数据数字转化为文字，方便绘图分析
telcom['SeniorCitizen'] = telcom['SeniorCitizen'].replace({1:'Yes',0:'No'})


# In[6]:


#对分类变量进行描述性分析
cate_col = telcom.nunique()[telcom.nunique()<5].keys().tolist()[0:-1]
churn_yes = telcom[cate_col][telcom['Churn']=='Yes']
churn_no = telcom[cate_col][telcom['Churn']=='No']

def plot_pie(a,b): 
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.pie(a,wedgeprops=dict(width=0.5),labeldistance =1.1,pctdistance = 0.7,labels=a.keys(),autopct='%.2f%%',colors = ['tomato', 'lightskyblue', 'goldenrod', 'green', 'y'])
    plt.title('流失用户',fontsize=13)
    plt.subplot(122)
    plt.pie(b,wedgeprops=dict(width=0.5),labels=b.keys(),autopct='%.2f%%',labeldistance = 1.1,pctdistance = 0.7,colors = ['tomato', 'lightskyblue', 'goldenrod', 'green', 'y'])
    plt.title('非流失用户',fontsize=13)
    
for i in cate_col:
    a = churn_yes[i].value_counts()
    b = churn_no[i].value_counts()
    plot_pie(a,b)
    plt.show()
    print('%s\n'.center(60) %(i+' -- 用户流失图解'))


# In[7]:


telcom['tenure'].describe()


# In[8]:


#将用户使用时间转化为分类变量分析，将流失与非流失用户做比较
bins = [0,12,24,48,72]
telcom['tenure'] = pd.cut(telcom['tenure'],bins,labels=['tenure0_12','tenure12_24','tenure24_48','tenure48_72'])
churn_tenure = telcom[['tenure','Churn']]

groups = churn_tenure.groupby(['Churn','tenure'])
groups['Churn'].count()

df_churn = pd.DataFrame({'No':groups['Churn'].count()['No'],'Yes':groups['Churn'].count()['Yes']})
df_churn.plot(kind='bar',stacked=True)
plt.xlabel('客户服务月份')
plt.ylabel('客户数量')
plt.ylim(0,2500)
plt.title('客户流失与服务月份的关系')


# In[9]:


#探索用户月消费、总消费、是否流失之间的关系
charge_yes = telcom[telcom['Churn']=='Yes'][['MonthlyCharges','TotalCharges']]
charge_no = telcom[telcom['Churn']=='No'][['MonthlyCharges','TotalCharges']]

fg = plt.figure(figsize=(10,8))
plt.scatter(charge_yes['MonthlyCharges'],charge_yes['TotalCharges'],label='churn-Yes',marker='.')
plt.scatter(charge_no['MonthlyCharges'],charge_no['TotalCharges'],label='churn-No',marker='.')
plt.xlabel('月消费')
plt.ylabel('总消费')
plt.legend()


# In[10]:


#探索用户月消费、总消费、使用时间之间的关系
tenure_charge = telcom[['MonthlyCharges','TotalCharges','tenure']]

charge0_12 = tenure_charge[tenure_charge['tenure']=='tenure0_12'][['MonthlyCharges','TotalCharges']]
charge12_24 = tenure_charge[tenure_charge['tenure']=='tenure12_24'][['MonthlyCharges','TotalCharges']]
charge24_48 = tenure_charge[tenure_charge['tenure']=='tenure24_48'][['MonthlyCharges','TotalCharges']]
charge48_72 = tenure_charge[tenure_charge['tenure']=='tenure48_72'][['MonthlyCharges','TotalCharges']]

fg = plt.figure(figsize=(10,8))
plt.scatter(charge0_12['MonthlyCharges'],charge0_12['TotalCharges'],label='tenure0_12',marker='.')
plt.scatter(charge12_24['MonthlyCharges'],charge12_24['TotalCharges'],label='tenure12_24',marker='.')
plt.scatter(charge24_48['MonthlyCharges'],charge24_48['TotalCharges'],label='tenure24_48',marker='.')
plt.scatter(charge48_72['MonthlyCharges'],charge48_72['TotalCharges'],label='tenure48_72',marker='.')
plt.xlabel('月消费')
plt.ylabel('总消费')
plt.legend()


# # 3. 特征工程

# In[11]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[12]:


#使用labelencoder处理具有两种类型的虚拟变量
cat_two = telcom.nunique()[telcom.nunique()==2].keys().tolist()
lab = LabelEncoder()
for i in cat_two:
    telcom[i] = lab.fit_transform(telcom[i])
#处理类型大于2的虚拟变量
cat_column = telcom.nunique()[telcom.nunique()<5 ].keys().tolist()
cat_more = [i for i in cat_column if i not in cat_two ]
telcom = pd.get_dummies(telcom,columns = cat_more)
#处理连续变量
num_col = telcom.nunique()[telcom.nunique()>5].keys().tolist()[1:]
std = StandardScaler()
te = std.fit_transform(telcom[num_col])
charge_std = pd.DataFrame(te,columns=num_col)
telcom = telcom.drop(columns=num_col)
telcom.reset_index(drop=True, inplace=True)
telcom = telcom.merge(charge_std,left_index=True,right_index=True,how = "left")
telcom_del = telcom.drop(columns='customerID')
telcom.head()



# # 4. 训练模型
# 

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


# In[14]:


#拆分数据为训练集和测试集
train_data = telcom_del.drop(columns='Churn')
train_target = telcom_del['Churn']
train_x,test_x,train_y,test_y = train_test_split(train_data,train_target,test_size=0.3,random_state=0)


# In[153]:


from sklearn.tree import DecisionTreeClassifier
dtmodel = DecisionTreeClassifier(max_leaf_nodes=8)
dtmodel.fit(train_x,train_y)
cross_val_score(knnmodel,train_data,train_target,cv=5)


# In[174]:


dtmodel_df = pd.DataFrame({'feature_num':dtmodel.feature_importances_.tolist(),'cate_column':train_x.columns.tolist()})
dtmodel_df.sort_values('feature_num',inplace=True,ascending=False)
dtmodel_series = pd.Series(dtmodel_df['feature_num'].tolist(),index=dtmodel_df['cate_column'].tolist())

plt.figure(figsize=(12,6))
dtmodel_series.plot(kind='bar')
plt.title('Feature Importance',fontsize=15)


# In[39]:


from sklearn.ensemble import RandomForestClassifier
rdfmodel = RandomForestClassifier()
rdfmodel.fit(train_x,train_y)
cross_val_score(rdfmodel,train_data,train_target,cv=5)


# In[175]:


rdfmodel_df = pd.DataFrame({'feature_num':rdfmodel.feature_importances_.tolist(),'cate_column':train_x.columns.tolist()})
rdfmodel_df.sort_values('feature_num' ,inplace=True,ascending=False)
rdfmodel_series = pd.Series(rdfmodel_df['feature_num'].tolist(),index=rdfmodel_df['cate_column'].tolist())

plt.figure(figsize=(12,6))
rdfmodel_series.plot(kind='bar')
plt.title('Feature Importance',fontsize=15)





# In[158]:


from sklearn import svm
linmodel = svm.LinearSVC()
linmodel.fit(train_x,train_y)
cross_val_score(linmodel,train_data,train_target,cv=5)


# In[176]:


lin_df = pd.DataFrame({'feature_num':linmodel.coef_[0],'cate_column':train_x.columns.tolist()})
lin_df.sort_values('feature_num',inplace=True,ascending=False)
lin_series = pd.Series(lin_df['feature_num'].tolist(),index=lin_df['cate_column'].tolist())

plt.figure(figsize=(12,6))
lin_series.plot(kind='bar')
plt.title('Feature Importance',fontsize=15)


# In[145]:


from sklearn import linear_model
lrmodel = linear_model.LogisticRegression()
lrmodel.fit(train_x,train_y)
cross_val_score(lrmodel,train_data,train_target,cv=5)


# In[177]:


lrmodel_df = pd.DataFrame({'feature_num':lrmodel.coef_[0],'cate_column':train_x.columns.tolist()})
lrmodel_df.sort_values('feature_num',inplace=True,ascending=False)
lrmodel_series = pd.Series(lrmodel_df['feature_num'].tolist(),index=lrmodel_df['cate_column'].tolist())

plt.figure(figsize=(12,6))
lrmodel_series.plot(kind='bar')
plt.title('Feature Importance',fontsize=15)

