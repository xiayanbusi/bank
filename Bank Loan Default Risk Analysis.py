import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
##1 read data, get general information##
data=pd.read_csv('C:\\Users\\hanfe\\Desktop\\Job\\projects\\application_data.csv')
# data basic information#
# print(data.head())
# print(data.info())
# print(data.describe())

# find percentage of missing values that higher than 40%
# null_pct = pd.DataFrame((data.isnull().sum())*100/data.shape[0]).reset_index()
# null_pct.columns = ['Column Name', 'Null Values Percentage']
# fig = plt.figure(figsize=(18,6))
# ax = sns.pointplot(x="Column Name",y="Null Values Percentage",data=null_pct,color='blue')
# plt.xticks(rotation =90,fontsize =7)
# ax.axhline(40, ls='--',color='red')
# plt.title("Percentage of Missing values in application data")
# plt.ylabel("Null Values PERCENTAGE")
# plt.xlabel("COLUMNS")
# plt.show()
##2.drop unwanted data##
# print(data.isnull().sum().sort_values(ascending=False)[:30])
# print(round(data.isnull().sum() / data.shape[0] * 100.00,2))
null_pct=round(data.isnull().sum() / data.shape[0] * 100.00,2)
# print(null_pct[null_pct.values>40].index)
data=data.drop(null_pct[null_pct.values>=40].index,axis=1)
# print(data.info())
pd.set_option('display.max_rows',None)
# print(data.nunique())



# Analyze & Delete Unnecessary Columns in data
# Source = data[["EXT_SOURCE_2","EXT_SOURCE_3","TARGET"]]
# source_corr = Source.corr()
# ax = sns.heatmap(source_corr,
#             xticklabels=source_corr.columns,
#             yticklabels=source_corr.columns,
#             annot = True,
#             cmap ="RdYlGn")
# plt.show()

# contact_col = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
#        'FLAG_PHONE', 'FLAG_EMAIL','TARGET']
# Contact_corr = data[contact_col].corr()
# fig = plt.figure(figsize=(8,8))
# ax = sns.heatmap(Contact_corr,
#             xticklabels=Contact_corr.columns,
#             yticklabels=Contact_corr.columns,
#             annot = True,
#             cmap ="RdYlGn",
#             linewidth=1)
# plt.show()
#mpute AMT_ANNUITY with median as the distribution is greatly skewed:
# plt.figure(figsize=(6,6))
# sns.kdeplot(data['AMT_ANNUITY'])
# plt.show()

# Impute AMT_GOODS_PRICE with mode as the distribution is closely similar:
# plt.figure(figsize=(6,6))
# sns.kdeplot(data['AMT_GOODS_PRICE'])
# plt.show()

# print(data['AMT_GOODS_PRICE'].isnull().sum())
statsDF=pd.DataFrame()
statsDF['AMT_GOODS_PRICE_mode']=data['AMT_GOODS_PRICE'].fillna(data['AMT_GOODS_PRICE'].mode()[0])
statsDF['AMT_GOODS_PRICE_mean']=data['AMT_GOODS_PRICE'].fillna(data['AMT_GOODS_PRICE'].mean())
statsDF['AMT_GOODS_PRICE_median']=data['AMT_GOODS_PRICE'].fillna(data['AMT_GOODS_PRICE'].median())
cols=['AMT_GOODS_PRICE_mode','AMT_GOODS_PRICE_mean','AMT_GOODS_PRICE_median']
# plt.figure(figsize=(18,10),dpi=90)
# plt.suptitle('Distribution of original data vs imputed data'.title())
# plt.subplot(221)
# sns.distplot(data['AMT_GOODS_PRICE'][pd.notnull(data['AMT_GOODS_PRICE'])]);
# for i in enumerate(cols):
#     plt.subplot(2,2,i[0]+2)
#     sns.distplot(statsDF[i[1]])
# plt.show()
# print(data['AMT_GOODS_PRICE'][pd.notnull(data['AMT_GOODS_PRICE'])])
# print(data['AMT_GOODS_PRICE'].mode()[0])
# print(data['AMT_GOODS_PRICE'].mean())
# print(data['AMT_GOODS_PRICE'].median())
data=data.drop(['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL','FLAG_DOCUMENT_2','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
           'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
           'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
           'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21','EXT_SOURCE_2','EXT_SOURCE_3'],axis=1)
print(data.shape)
# print(data.info())
##After deleting unnecessary columns, there are 46 columns remaining in data ##

## 3.fill null values##
#3.1 Converting Negative days to positive days#
# print(data['DAYS_BIRTH'][:10])
print(data.isnull().sum())
date_col=['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']
for col in date_col:
    data[col]=abs(data[col])
# print(data['DAYS_BIRTH'][:10])

#3.2 Binning Numerical Columns to create a categorical column
print(data['AMT_INCOME_TOTAL'].nlargest(5))
print(data['AMT_INCOME_TOTAL'].nsmallest(5))
bin_number=np.log2(data.shape[0])
print(f'Suggested bins number : {bin_number}')
data['AMT_INCOME_TOTAL']=data['AMT_INCOME_TOTAL']/100000
bins=np.arange(0,12,1)
slot=['0-100k','100-200k','200-300k','300-400k','400-500k','500-600k','600-700k','700-800k','800-900k','900-1M','1M above']
data['AMT_INCOME_RANGE']=pd.cut(data['AMT_INCOME_TOTAL'],bins,labels=slot)
print(data['AMT_INCOME_RANGE'].head())
print(data['AMT_INCOME_RANGE'].value_counts(normalize=True)*100)
#More than 50% loan applicants have income amount in the range of 100K-200K. Almost 92% loan applicants have income less than 300K#
#3.3 age
data['AGE']=data['DAYS_BIRTH'] // 365
bins = [0,20,30,40,50,100]
slots = ['0-20','20-30','30-40','40-50','50 above']
data['AGE_GROUP']=pd.cut(data['AGE'],bins=bins,labels=slots)
print(data['AGE_GROUP'].tail())
print(data['AGE_GROUP'].value_counts(normalize=True)*100)
print('*'*50)
print(data['AGE_GROUP'].value_counts(normalize=True))
print('*'*50)
print(data['AGE_GROUP'].value_counts())
#3.4 Creating bins for Employement Time
data['YEARS_EMPLOYED'] = data['DAYS_EMPLOYED'] // 365
bins = [0,5,10,20,30,40,50,60,150]
slots = ['0-5','5-10','10-20','20-30','30-40','40-50','50-60','60 above']

data['EMPLOYMENT_YEAR']=pd.cut(data['YEARS_EMPLOYED'],bins=bins,labels=slots)
print(data['EMPLOYMENT_YEAR'].value_counts(normalize=True)*100)
print(data.nunique().sort_values())
#3.5 Conversion of Object and Numerical columns to Categorical Columns
categorical_columns = ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','LIVE_CITY_NOT_WORK_CITY',
                       'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','REG_REGION_NOT_WORK_REGION',
                       'LIVE_REGION_NOT_WORK_REGION','REGION_RATING_CLIENT','WEEKDAY_APPR_PROCESS_START',
                       'REGION_RATING_CLIENT_W_CITY'
                      ]
for col in categorical_columns:
    data[col] =pd.Categorical(data[col])
print(data.info())
#4. Null Value Data Imputation
# checking the null value % of each column in applicationDF dataframe
print(round(data.isnull().sum() / data.shape[0] * 100.00,2))
print(data['NAME_TYPE_SUITE'].describe())
data['NAME_TYPE_SUITE'].fillna((data['NAME_TYPE_SUITE'].mode()[0]),inplace = True)
data['OCCUPATION_TYPE'] = data['OCCUPATION_TYPE'].cat.add_categories('Unknown')
data['OCCUPATION_TYPE'].fillna('Unknown', inplace =True)
print(data[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',
               'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
               'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].describe())
amount = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']

for col in amount:
    data[col].fillna(data[col].median(),inplace = True)
# checking the null value % of each column in previousDF dataframe
print(round(data.isnull().sum() / data.shape[0] * 100.00,2))
print(data['EMPLOYMENT_YEAR'].describe())
print(data['EMPLOYMENT_YEAR'].unique())
data['EMPLOYMENT_YEAR'].fillna(data['EMPLOYMENT_YEAR'].mode()[0],inplace=True)

#4. data analysis
Imbalance = data["TARGET"].value_counts().reset_index()

plt.figure(figsize=(10,4))
x= ['Repayer','Defaulter']
sns.barplot(x,"TARGET",data = Imbalance,palette= ['g','r'])
plt.xlabel("Loan Repayment Status")
plt.ylabel("Count of Repayers & Defaulters")
plt.title("Imbalance Plotting")
# plt.show()

count_0 = Imbalance.iloc[0]["TARGET"]
count_1 = Imbalance.iloc[1]["TARGET"]
count_0_perc = round(count_0/(count_0+count_1)*100,2)
count_1_perc = round(count_1/(count_0+count_1)*100,2)

print('Ratios of imbalance in percentage with respect to Repayer and Defaulter datas are: %.2f and %.2f'%(count_0_perc,count_1_perc))
print('Ratios of imbalance in relative with respect to Repayer and Defaulter datas is %.2f : 1 (approx)'%(count_0/count_1))

#4.1 Numeric Variables Analysis
print(data.columns)
cols_for_correlation = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                        'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
                        'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
                        'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
                        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
                        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE',
                        'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3',
                        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                        'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

# Getting the top 10 correlation for the Repayers data
Repayer_df = data.loc[data['TARGET']==0, cols_for_correlation] # Repayers
Defaulter_df =data.loc[data['TARGET']==1, cols_for_correlation] # Defaulters
corr_repayer = Repayer_df.corr()
print(corr_repayer)
corr_repayer = corr_repayer.where(np.triu(np.ones(corr_repayer.shape),k=1).astype(bool))
print(corr_repayer)
corr_df_repayer = corr_repayer.unstack().reset_index()
corr_df_repayer.columns =['VAR1','VAR2','Correlation']
corr_df_repayer.dropna(subset = ["Correlation"], inplace = True)
corr_df_repayer["Correlation"]=corr_df_repayer["Correlation"].abs()
corr_df_repayer.sort_values(by='Correlation', ascending=False, inplace=True)
print(corr_df_repayer.head(10))
fig = plt.figure(figsize=(12,12))
ax = sns.heatmap(Repayer_df.corr(), cmap="RdYlGn",annot=False,linewidth =1)
# plt.show()

# Getting the top 10 correlation for the Defaulter data
corr_Defaulter = Defaulter_df.corr()
corr_Defaulter = corr_Defaulter.where(np.triu(np.ones(corr_Defaulter.shape),k=1).astype(bool))
corr_df_Defaulter = corr_Defaulter.unstack().reset_index()
corr_df_Defaulter.columns =['VAR1','VAR2','Correlation']
corr_df_Defaulter.dropna(subset = ["Correlation"], inplace = True)
corr_df_Defaulter["Correlation"]=corr_df_Defaulter["Correlation"].abs()
corr_df_Defaulter.sort_values(by='Correlation', ascending=False, inplace=True)
print(corr_df_Defaulter.head(10))

###Inferences:
#Credit amount is highly correlated with amount of goods price which is same as repayers.
#But the loan annuity correlation with credit amount has slightly reduced in defaulters(0.75) when compared to repayers(0.77)
#We can also see that repayers have high correlation in number of days employed(0.62) when compared to defaulters(0.58).
#There is a severe drop in the correlation between total income of the client and the credit amount(0.038) amongst defaulters whereas it is 0.342 among repayers.
#Days_birth and number of children correlation has reduced to 0.259 in defaulters when compared to 0.337 in repayers.
#There is a slight increase in defaulted to observed count in social circle among defaulters(0.264) when compared to repayers(0.254)

# function for plotting repetitive countplots in univariate categorical analysis on applicationDF
# This function will create two subplots:
# 1. Count plot of categorical column w.r.t TARGET;
# 2. Percentage of defaulters within column

def univariate_categorical(feature, ylog=False, label_rotation=False, horizontal_layout=True):
    temp = data[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = data[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    cat_perc["TARGET"] = cat_perc["TARGET"] * 100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    if (horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 24))

    # 1. Subplot 1: Count plot of categorical column
    # sns.set_palette("Set2")
    s = sns.countplot(ax=ax1,
                      x=feature,
                      data=data,
                      hue="TARGET",
                      order=cat_perc[feature],
                      palette=['g', 'r'])

    # Define common styling
    ax1.set_title(feature, fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})
    ax1.legend(['Repayer', 'Defaulter'])

    # If the plot is not readable, use the log scale.
    if ylog:
        ax1.set_yscale('log')
        ax1.set_ylabel("Count (log)", fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})

    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    # 2. Subplot 2: Percentage of defaulters within the categorical column
    s = sns.barplot(ax=ax2,
                    x=feature,
                    y='TARGET',
                    order=cat_perc[feature],
                    data=cat_perc,
                    palette='Set2')

    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.ylabel('Percent of Defaulters [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(feature + " Defaulter %", fontdict={'fontsize': 15, 'fontweight': 5, 'color': 'Blue'})

    plt.show();


# function for plotting repetitive countplots in bivariate categorical analysis

def bivariate_bar(x, y, df, hue, figsize):
    plt.figure(figsize=figsize)
    sns.barplot(x=x,
                y=y,
                data=df,
                hue=hue,
                palette=['g', 'r'])

    # Defining aesthetics of Labels and Title of the plot using style dictionaries
    plt.xlabel(x, fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})
    plt.ylabel(y, fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})
    plt.title(col, fontdict={'fontsize': 15, 'fontweight': 5, 'color': 'Blue'})
    plt.xticks(rotation=90, ha='right')
    plt.legend(labels=['Repayer', 'Defaulter'])
    plt.show()


def bivariate_rel(x, y, data, hue, kind, palette, legend, figsize):
    plt.figure(figsize=figsize)
    sns.relplot(x=x,
                y=y,
                data=data,
                hue="TARGET",
                kind=kind,
                palette=['g', 'r'],
                legend=False)
    plt.legend(['Repayer', 'Defaulter'])
    plt.xticks(rotation=90, ha='right')
    plt.show()


# function for plotting repetitive countplots in univariate categorical analysis on the merged df

def univariate_merged(col, df, hue, palette, ylog, figsize):
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=col,
                       data=df,
                       hue=hue,
                       palette=palette,
                       order=df[col].value_counts().index)

    if ylog:
        plt.yscale('log')
        plt.ylabel("Count (log)", fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})
    else:
        plt.ylabel("Count", fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})

    plt.title(col, fontdict={'fontsize': 15, 'fontweight': 5, 'color': 'Blue'})
    plt.legend(loc="upper right")
    plt.xticks(rotation=90, ha='right')

    plt.show()
# Checking the contract type based on loan repayment status
univariate_categorical('NAME_CONTRACT_TYPE',True)
# Checking the type of Gender on loan repayment status
univariate_categorical('CODE_GENDER')
# Checking if owning a car is related to loan repayment status
univariate_categorical('FLAG_OWN_CAR')
# Analyzing Housing Type based on loan repayment status
univariate_categorical("NAME_HOUSING_TYPE",True,True,True)
# Analyzing Family status based on loan repayment status
univariate_categorical("NAME_FAMILY_STATUS",False,True,True)
# Analyzing Education Type based on loan repayment status
univariate_categorical("NAME_EDUCATION_TYPE",True,True,True)
# Analyzing Income Type based on loan repayment status
univariate_categorical("NAME_INCOME_TYPE",True,True,False)
# Analyzing Region rating where applicant lives based on loan repayment status
univariate_categorical("REGION_RATING_CLIENT",False,False,True)
# Analyzing Occupation Type where applicant lives based on loan repayment status
univariate_categorical("OCCUPATION_TYPE",False,True,False)
# Checking Loan repayment status based on Organization type
univariate_categorical("ORGANIZATION_TYPE",True,True,False)
# Analyzing Flag_Doc_3 submission status based on loan repayment status
univariate_categorical("FLAG_DOCUMENT_3",False,False,True)
# Analyzing Age Group based on loan repayment status
univariate_categorical("AGE_GROUP",False,False,True)
# Analyzing Employment_Year based on loan repayment status
univariate_categorical("EMPLOYMENT_YEAR",False,False,True)
# Analyzing Amount_Credit based on loan repayment status
univariate_categorical("AMT_CREDIT_RANGE",False,False,False)
# Analyzing Amount_Income Range based on loan repayment status
univariate_categorical("AMT_INCOME_RANGE",False,False,False)
# Analyzing Number of children based on loan repayment status
univariate_categorical("CNT_CHILDREN",True)
# Analyzing Number of family members based on loan repayment status
univariate_categorical("CNT_FAM_MEMBERS",True, False, False)
#4.3 Categorical Bi/Multivariate Analysis
data.groupby('NAME_INCOME_TYPE')['AMT_INCOME_TOTAL'].describe()
# Income type vs Income Amount Range
bivariate_bar("NAME_INCOME_TYPE","AMT_INCOME_TOTAL",applicationDF,"TARGET",(18,10))
