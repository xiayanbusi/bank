import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('C:\\Users\\hanfe\\Desktop\\Job\\projects\\Churn_Modelling\\Churn_Modelling.csv')
# data basic information#
# print(data.head())
print(data.info())
# print(data.describe())
# print(data.isnull().sum())
print(round(data.isnull().sum() / data.shape[0] * 100.00,2))
print('^'*50)
print(data.nunique())
data=data.drop(['Surname','RowNumber'],axis=1)
print(data.head())
print(data.groupby('Geography')['CustomerId'].count())
print(data.groupby('Gender')['CustomerId'].count())
gender_dis=data.groupby('Gender')['CustomerId'].count()
#gender percentage caculation#
gender_pct=gender_dis.values/sum(gender_dis.values)
gender_pct='%.2f%%'%(gender_pct[0]*100),'%.2f%%'%(gender_pct[1]*100)
#gender bar plot#
# plt.figure(figsize=(12,7),dpi=90)
# ax=plt.bar(gender_dis.index,gender_dis.values)
# labels=(4543,5457)
# plt.bar_label(ax)
# plt.bar_label(ax,labels=gender_pct,label_type='center',color='white')
# plt.title('bank customers by gender'.title())
# plt.show()

geography_pct=data.groupby('Geography')['CustomerId'].count().values/sum(data.groupby('Geography')['CustomerId'].count())
geography_pct='%.2f%%'%(geography_pct[0]*100),'%.2f%%'%(geography_pct[1]*100),'%.2f%%'%(geography_pct[2]*100)
print(geography_pct)
print(data['Balance'].nlargest(10))
print(data[data['Balance']==0.0]['CustomerId'].count())
# account balance equals 0 by gender #
print(data[data['Balance']==0.0].groupby('Gender')['CustomerId'].count())
# account balance equals 0 by geography#
print(data[data['Balance']==0.0].groupby('Geography')['CustomerId'].count())
# bank account equals 0 by age distribution( top 20)#
# print(data[data['Balance']==0.0].groupby('Age')['CustomerId'].count().sort_values(ascending=False)[:20])
# bank accunt equeals 0 by age distribution plot#
# plt.plot(data[data['Balance']==0.0].groupby('Age')['CustomerId'].count().values)
# plt.title('bank accunt equeals 0 by age distribution plot'.title())
# plt.xlabel('Age')
# plt.ylabel('Accounts Count')
# plt.show()
print(data.groupby('Exited')['CustomerId'].count())
print(data.groupby('HasCrCard')['CustomerId'].count())
print(data.groupby('IsActiveMember')['CustomerId'].count())
print(data['EstimatedSalary'].nlargest(10))
print(data['EstimatedSalary'].nsmallest(10))
print(data['EstimatedSalary'].describe())
# plt.hist(data['EstimatedSalary'].values,bins=30)
# plt.show()
#histogram of account banlance#
# plt.hist(data['Balance'].values,bins=10,label='Account Balance')
# plt.hist(data['EstimatedSalary'].values,bins=10,label='Estimated Salary')
# plt.legend()
# plt.show()
#fraud analysis: account balance more than 75 percentile,but Esitmated salary lower than 25 percentile #
print(data['Balance'].describe())
print(data[(data['Balance']>76485.889288) & (data['EstimatedSalary']<51002.110000)][['CustomerId','CreditScore','Age','Balance','EstimatedSalary']])
print(data[(data['Balance']>127644.240000) & (data['EstimatedSalary']<51002.110000) ][['CustomerId','CreditScore','Age','Balance','EstimatedSalary']])
# # histogram of creditscore#
# data['CreditScore'].plot(kind='hist')
# plt.show()
print(data[(data['Balance']>127644.240000) & (data['EstimatedSalary']<51002.110000) & (data['CreditScore']<=500)][['CustomerId','CreditScore','Age','Balance','EstimatedSalary']])
print(data[(data['Balance']>127644.240000) & (data['EstimatedSalary']<51002.110000) & (data['CreditScore']<=500)][['CustomerId','CreditScore','Age','Balance','EstimatedSalary']].count())
# data['Age'].plot(kind='hist')
#
# plt.show()
# data['Tenure'].plot(kind='hist')
# plt.show()
#pie chart for excited and retained customers#
# labels='Retained','Exited'
# print(data.groupby('Exited')['CustomerId'].count())
# data.groupby('Exited')['CustomerId'].count().plot(kind='pie',labels=labels,autopct='%1.1f%%',
#         shadow=True, startangle=90)
# plt.show()

# fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
# sns.countplot(x='Geography', hue = 'Exited',data =data, ax=axarr[0][0])
# sns.countplot(x='Gender', hue = 'Exited',data =data, ax=axarr[0][1])
# sns.countplot(x='HasCrCard', hue = 'Exited',data =data, ax=axarr[1][0])
# sns.countplot(x='IsActiveMember', hue = 'Exited',data =data, ax=axarr[1][1])

# Relations based on the continuous data attributes
# fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
# sns.boxplot(y='CreditScore', x='Exited', hue='Exited', data=data, ax=axarr[0][0])
# sns.boxplot(y='Age', x='Exited', hue='Exited', data=data, ax=axarr[0][1])
# sns.boxplot(y='Tenure', x='Exited', hue='Exited', data=data, ax=axarr[1][0])
# sns.boxplot(y='Balance', x='Exited', hue='Exited', data=data, ax=axarr[1][1])
# sns.boxplot(y='NumOfProducts', x='Exited', hue='Exited', data=data, ax=axarr[2][0])
# sns.boxplot(y='EstimatedSalary', x='Exited', hue='Exited', data=data, ax=axarr[2][1])
# plt.show()
# meachine learning#
df_train = data.sample(frac=0.8,random_state=200)
df_test = data.drop(df_train.index)
df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
df_train['TenureByAge'] = df_train.Tenure/(df_train.Age)
df_train['CreditScoreGivenAge'] = df_train.CreditScore/(df_train.Age)
# sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df_train)
# plt.ylim(-1, 5)
# plt.show()
#Data prep for model fitting
continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
df_train = df_train[['Exited'] + continuous_vars + cat_vars]
print(df_train.head())
df_train.loc[df_train.HasCrCard == 0, 'HasCrCard'] = -1
df_train.loc[df_train.IsActiveMember == 0, 'IsActiveMember'] = -1
print(df_train.head())
lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if (df_train[i].dtype ==str or df_train[i].dtype == object):
        for j in df_train[i].unique():
            df_train[i+'_'+j] = np.where(df_train[i] == j,1,-1)
        remove.append(i)
df_train = df_train.drop(remove, axis=1)
print(df_train.head())
# minMax scaling the continuous variables
minVec = df_train[continuous_vars].min().copy()
maxVec = df_train[continuous_vars].max().copy()
df_train[continuous_vars] = (df_train[continuous_vars]-minVec)/(maxVec-minVec)
print(df_train.head())
# data prep pipeline for test data
# def DfPrepPipeline(df_predict,df_train_Cols,minVec,maxVec):
#     # Add new features
#     df_predict['BalanceSalaryRatio'] = df_predict.Balance/df_predict.EstimatedSalary
#     df_predict['TenureByAge'] = df_predict.Tenure/(df_predict.Age - 18)
#     df_predict['CreditScoreGivenAge'] = df_predict.CreditScore/(df_predict.Age - 18)
#     # Reorder the columns
#     continuous_vars = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','BalanceSalaryRatio',
#                    'TenureByAge','CreditScoreGivenAge']
#     cat_vars = ['HasCrCard','IsActiveMember',"Geography", "Gender"]
#     df_predict = df_predict[['Exited'] + continuous_vars + cat_vars]
#     # Change the 0 in categorical variables to -1
#     df_predict.loc[df_predict.HasCrCard == 0, 'HasCrCard'] = -1
#     df_predict.loc[df_predict.IsActiveMember == 0, 'IsActiveMember'] = -1
#     # One hot encode the categorical variables
#     lst = ["Geography", "Gender"]
#     remove = list()
#     for i in lst:
#         for j in df_predict[i].unique():
#             df_predict[i+'_'+j] = np.where(df_predict[i] == j,1,-1)
#         remove.append(i)
#     df_predict = df_predict.drop(remove, axis=1)
#     # Ensure that all one hot encoded variables that appear in the train data appear in the subsequent data
#     L = list(set(df_train_Cols) - set(df_predict.columns))
#     for l in L:
#         df_predict[str(l)] = -1
#     # MinMax scaling coontinuous variables based on min and max from the train data
#     df_predict[continuous_vars] = (df_predict[continuous_vars]-minVec)/(maxVec-minVec)
#     # Ensure that The variables are ordered in the same way as was ordered in the train set
#     df_predict = df_predict[df_train_Cols]
#     return df_predict

contact_col = ['CreditScore', 'Age','Gender','Geography', 'Balance',
       'EstimatedSalary', 'Tenure']
Contact_corr = data[contact_col].corr()
fig = plt.figure(figsize=(8,8))
ax = sns.heatmap(Contact_corr,
            xticklabels=Contact_corr.columns,
            yticklabels=Contact_corr.columns,
            annot = True,
            cmap ="RdYlGn",
            linewidth=1)
plt.show()