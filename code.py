#!/usr/bin/env python
# coding: utf-8

# ## Name: Saad Ahmed
# ## Roll_No: 19I-1705
# ## Section: DS-N
# ## Fundamentals of Data Science (Semester Project)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # a. Data Reading:

# In[2]:


df = pd.read_csv('Blood_Pressure_data.csv')


# In[3]:


pd.set_option('display.max_columns',None)


# # b. Data Cleaning:

# In[4]:


df.head(3)


# In[5]:


df.drop(['id','patient_no','weight','medical_specialty'],axis=1,inplace = True)


# In[6]:


df.head(2)


# ##       1. Handling Missing Values:

# In[7]:


df['cast'].value_counts()


# In[8]:


df = df[df['cast']!='?']


# In[9]:


df.head()


# In[10]:


df['payer_code'].value_counts()


# In[11]:


# Since Most of the Values are missing. So, I am dropping this column.
df.drop('payer_code',axis=1,inplace=True)


# In[12]:


df[df['diag_1']=='?'].describe()


# In[13]:


df=df[df['diag_1']!='?']


# In[14]:


df[df['diag_2']=='?'].describe()


# In[15]:


df=df[df['diag_2']!='?']


# In[16]:


df[df['diag_3']=='?'].describe()


# In[17]:


df=df[df['diag_3']!='?']


# In[18]:


# Keeping Records of Columns for which I have to detect Outliers
outlier_detection=['number_outpatient','number_emergency','number_inpatient']


# In[19]:


df[df['metformin-pioglitazone']=='?'].describe()


# In[20]:


df['gender'].unique()


# In[21]:


df=df[df['gender']!='Unknown/Invalid']


# In[22]:


df.head()


# In[23]:


df[df.duplicated()].describe()


# #### ----------------------------Missing Values have been dealt--------------------------

# ## 2. Data Mapping/Encoding

# In[24]:


df.dtypes


# ### To map/encode the dataset, we need to understand it's features/columns. I found this same dataset online at https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip
# #### In this they also clearly mentioned that what the numbers denote in 'admission_type_id', 'discharge_disposition_type_id', 'admission_source_id', which means they specified that what these numbers in these columns tell us about.

# ### 'discharge_disposition_id'
# 1.	Discharged to home
# 2.	Discharged/transferred to another short term hospital
# 3.	Discharged/transferred to SNF
# 4.	Discharged/transferred to ICF
# 5.	Discharged/transferred to another type of inpatient care institution
# 6.	Discharged/transferred to home with home health service
# 7.	Left AMA
# 8.	Discharged/transferred to home under care of Home IV provider
# 9.	Admitted as an inpatient to this hospital
# 10.	Neonate discharged to another hospital for neonatal aftercare
# 11.	Expired
# 12.	Still patient or expected to return for outpatient services
# 13.	Hospice / home
# 14.	Hospice / medical facility
# 15.	Discharged/transferred within this institution to Medicare approved swing bed
# 16.	Discharged/transferred/referred another institution for outpatient services
# 17.	Discharged/transferred/referred to this institution for outpatient services
# 18.	NULL
# 19.	Expired at home. Medicaid only, hospice.
# 20.	Expired in a medical facility. Medicaid only, hospice.
# 21.	Expired, place unknown. Medicaid only, hospice.
# 22.	Discharged/transferred to another rehab fac including rehab units of a hospital .
# 23.	Discharged/transferred to a long term care hospital.
# 24.	Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.
# 25.	Not Mapped
# 26.	Unknown/Invalid
# 30.	Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere
# 27.	Discharged/transferred to a federal health care facility.
# 28.	Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital
# 29.	Discharged/transferred to a Critical Access Hospital (CAH).
# 

# ##### In the above given list 1,6,8,13 have approx same meaning/purpose (discharged to Home)
# ##### Similarly, 11,19,20,21 have same meanings (patient is expired)
# ##### Similarly: 2,3,4,5,16,17,22,23,24,27,28,29,30 (transferred/referred to other hospital/facility)
# ##### 25,26 (NaN Value)

# In[25]:


df['discharge_disposition_id'].unique()


# In[26]:


ar=[ 1,  3,  6,  2,  5, 11,  7, 25, 10,  4, 14, 18,  8, 13, 12, 16, 17,
       22, 23,  9, 20, 15, 24, 28, 19, 27]


# In[27]:


a=np.array(ar)


# In[28]:


(a.sort())


# In[29]:


print(a)


# In[30]:


discharge={1:1,6:1,8:1,13:1,11:11,19:11,20:11,21:11,2:2,3:2,4:2,5:2,16:2,17:2,22:2,23:2,24:2,27:2,28:2,29:2,30:2,7:7,9:9,10:10,12:12,14:14,15:15,18:18,25:25}


# 

# In[31]:


df['discharge_disposition_id'].value_counts()


# In[32]:


df['discharge_disposition_id']=df['discharge_disposition_id'].map(discharge)


# In[33]:


df['discharge_disposition_id'].value_counts()


# ## Similarly admission_source_id is mapped according to this criteria:
# 
# 1.	 Physician Referral
# 2.	Clinic Referral
# 3.	HMO Referral
# 4.	Transfer from a hospital
# 5.	 Transfer from a Skilled Nursing Facility (SNF)
# 6.	 Transfer from another health care facility
# 7.	 Emergency Room
# 8.	 Court/Law Enforcement
# 9.	 Not Available
# 10.	 Transfer from critial access hospital
# 11.	Normal Delivery
# 12.	 Premature Delivery
# 13.	 Sick Baby
# 14.	 Extramural Birth
# 15.	Not Available
# 17.	NULL
# 18.	 Transfer From Another Home Health Agency
# 19.	Readmission to Same Home Health Agency
# 20.	 Not Mapped
# 21.	Unknown/Invalid
# 22.	 Transfer from hospital inpt/same fac reslt in a sep claim
# 23.	 Born inside this hospital
# 24.	 Born outside this hospital
# 25.	 Transfer from Ambulatory Surgery Center
# 26.	Transfer from Hospice
# 

# #### Following sets will be made according to the above map:
# ##### {1,2,3},{4,5,6,10,17,21,24,25},{9,15,16,19,20},{11,12,13,14,22}

# In[34]:


df['admission_source_id'].unique()


# In[35]:


admm=np.array([ 7,  2,  4,  1,  5,  6, 20,  3, 17,  8,  9, 14, 10, 22, 11, 25, 13])
admm.sort()
print(admm)


# In[36]:


adm={1:1,2:1,3:1,4:4,5:4,6:4,10:4,17:4,21:4,24:4,25:4,9:9,15:9,16:9,19:9,20:9,11:11,12:11,13:11,14:11,22:11,7:7,8:8}


# In[37]:


df['admission_source_id']=df['admission_source_id'].map(adm)


# In[38]:


df


# In[39]:


df['cast'].unique()


# In[40]:


from sklearn.preprocessing import LabelEncoder


# In[41]:


le=LabelEncoder()


# In[42]:


df['cast']=le.fit_transform(df['cast'])


# In[43]:


df['age group'].unique()


# In[44]:


agegrp={'[10-20)':1, '[20-30)':2, '[30-40)':3, '[40-50)':5, '[50-60)':6, '[60-70)':7,
       '[70-80)':8, '[80-90)':9, '[90-100)':10, '[0-10)':0}


# In[45]:


df['age group']=df['age group'].map(agegrp)


# In[46]:


gender={'Male':1,'Female':0}
df['gender']=df['gender'].map(gender)


# In[47]:


df['max_glu_serum'].unique()


# In[48]:


mgs = {'None':0, '>300':2, 'Norm':1, '>200':2}
# There could be 3 Scenarios: 
# 1. None 
# 2. It could be Normal
#3. It could be Abnormal


# In[49]:


df['max_glu_serum']=df['max_glu_serum'].map(mgs)


# In[50]:


plt.hist(df['max_glu_serum'],color=['red'])
plt.xlabel('Max Glu Serum',labelpad=10)
plt.title('Occurences of Max_Glu_Serum')
plt.ylabel('Frequency')
plt.show()


# In[51]:


df['A1Cresult'].unique()


# In[52]:


#A1Cresult would be treated the same way as max_glu_serum is treated
a1={'None':0, '>7':1, '>8':1, 'Norm':2}
df['A1Cresult']=df['A1Cresult'].map(a1)


# In[53]:


df['metformin'].unique()


# In[54]:


# These next columns from now on are the drug names given to the patients or these are the levels of the drugs in patient's body
# I would classify 'No' & 'Steady' as same that is I would map them as 0
# While I would be considering 'Up' & 'Down' which means that they are not Normal and I will be classifying them as 1
drug={'No':0, 'Steady':0, 'Up':1, 'Down':1}


# In[55]:


df.columns


# In[56]:


columns_names=[ 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']
# These are the columns which I would be mapping now


# In[57]:


for c in columns_names:
    df[c]=df[c].map(drug)


# In[58]:


df.head()


# In[59]:


df.change.unique()


# In[60]:


change = {'No':0,'Ch':1}
df['change']=df['change'].map(change)


# In[61]:


df['Med'].unique()


# In[62]:


Med={'No':0,'Yes':1}
df['Med']=df['Med'].map(Med)


# In[63]:


df.head()


# In[64]:


df['label'].unique()


# In[65]:


# As we are prediciting that will it happen or not. So, I'll classify label as 0 and 1
label={'>5':1, 'NO':0, '<30':1}
df['label']=df['label'].map(label)


# In[66]:


df


# Now the Columns which are left behind are diag_1, diag_2, diag_3
#  These diag_1,2,3 are basically the diagnosis which are made by the doctors
#  These numbers for example 276, V27 etc are the ICD (International Code of Diseases).
#  In ICD, there are certain ranges for different diseases. You can visit this link to check ranges : https://icd.codes/icd9cm

# In[67]:


df.loc[df['diag_1'].str.contains('V'),['diag_1']]=18
df.loc[df['diag_1'].str.contains('E')==True,['diag_1']]=19
df['diag_1']=df['diag_1'].astype(float)
for index in df.index:
    if df ['diag_1'][index]>=1 and df ['diag_1'][index]<140:
        df ['diag_1'][index]=1
    elif df ['diag_1'][index]>=140 and df ['diag_1'][index]<240:
        df ['diag_1'][index]=2
    elif df ['diag_1'][index]>=240 and df ['diag_1'][index]<280:
        df ['diag_1'][index]=3
    elif df ['diag_1'][index]>=280 and df ['diag_1'][index]<290:
        df ['diag_1'][index]=4
    elif df ['diag_1'][index]>=290 and df ['diag_1'][index]<320:
        df ['diag_1'][index]=5
    elif df ['diag_1'][index]>=320 and df ['diag_1'][index]<390:
        df ['diag_1'][index]=6
    elif df ['diag_1'][index]>=390 and df ['diag_1'][index]<460:
        df ['diag_1'][index]=7
    elif df ['diag_1'][index]>=460 and df ['diag_1'][index]<520:
        df ['diag_1'][index]=8
    elif df ['diag_1'][index]>=520 and df ['diag_1'][index]<580:
        df ['diag_1'][index]=9
    elif df ['diag_1'][index]>=580 and df ['diag_1'][index]<630:
        df ['diag_1'][index]=10
    elif df ['diag_1'][index]>=630 and df ['diag_1'][index]<680:
        df ['diag_1'][index]=11
    elif df ['diag_1'][index]>=680 and df ['diag_1'][index]<710:
        df ['diag_1'][index]=12
    elif df ['diag_1'][index]>=710 and df ['diag_1'][index]<740:
        df ['diag_1'][index]=13
    elif df ['diag_1'][index]>=740 and df ['diag_1'][index]<760:
        df ['diag_1'][index]=14
    elif df ['diag_1'][index]>=760 and df ['diag_1'][index]<780:
        df ['diag_1'][index]=15
    elif df ['diag_1'][index]>=780 and df ['diag_1'][index]<800:
        df ['diag_1'][index]=16
    elif df ['diag_1'][index]>=800 and df ['diag_1'][index]<1000:
        df ['diag_1'][index]=17


# In[68]:


df.loc[df['diag_2'].str.contains('V')==True,['diag_2']]=18

df.loc[df['diag_2'].str.contains('E')==True,['diag_2']]=19

df['diag_2']=df['diag_2'].astype(float)

for index in df.index:
    if df ['diag_2'][index]>=1 and df ['diag_2'][index]<140:
        df ['diag_2'][index]=1
    elif df ['diag_2'][index]>=140 and df ['diag_2'][index]<240:
        df ['diag_2'][index]=2
    elif df ['diag_2'][index]>=240 and df ['diag_2'][index]<280:
        df ['diag_2'][index]=3
    elif df ['diag_2'][index]>=280 and df ['diag_2'][index]<290:
        df ['diag_2'][index]=4
    elif df ['diag_2'][index]>=290 and df ['diag_2'][index]<320:
        df ['diag_2'][index]=5
    elif df ['diag_2'][index]>=320 and df ['diag_2'][index]<390:
        df ['diag_2'][index]=6
    elif df ['diag_2'][index]>=390 and df ['diag_2'][index]<460:
        df ['diag_2'][index]=7
    elif df ['diag_2'][index]>=460 and df ['diag_2'][index]<520:
        df ['diag_2'][index]=8
    elif df ['diag_2'][index]>=520 and df ['diag_2'][index]<580:
        df ['diag_2'][index]=9
    elif df ['diag_2'][index]>=580 and df ['diag_2'][index]<630:
        df ['diag_2'][index]=10
    elif df ['diag_2'][index]>=630 and df ['diag_2'][index]<680:
        df ['diag_2'][index]=11
    elif df ['diag_2'][index]>=680 and df ['diag_2'][index]<710:
        df ['diag_2'][index]=12
    elif df ['diag_2'][index]>=710 and df ['diag_2'][index]<740:
        df ['diag_2'][index]=13
    elif df ['diag_2'][index]>=740 and df ['diag_2'][index]<760:
        df ['diag_2'][index]=14
    elif df ['diag_2'][index]>=760 and df ['diag_2'][index]<780:
        df ['diag_2'][index]=15
    elif df ['diag_2'][index]>=780 and df ['diag_2'][index]<800:
        df ['diag_2'][index]=16
    elif df ['diag_2'][index]>=800 and df ['diag_2'][index]<1000:
        df ['diag_2'][index]=17


# In[69]:


df.loc[df['diag_3'].str.contains('V')==True,['diag_3']]=18
df.loc[df['diag_3'].str.contains('E')==True,['diag_3']]=19
df['diag_3']=df['diag_3'].astype(float)
for index in df.index:
    if df ['diag_3'][index]>=1 and df ['diag_3'][index]<140:
        df ['diag_3'][index]=1
    elif df ['diag_3'][index]>=140 and df ['diag_3'][index]<240:
        df ['diag_3'][index]=2
    elif df ['diag_3'][index]>=240 and df ['diag_3'][index]<280:
        df ['diag_3'][index]=3
    elif df ['diag_3'][index]>=280 and df ['diag_3'][index]<290:
        df ['diag_3'][index]=4
    elif df ['diag_3'][index]>=290 and df ['diag_3'][index]<320:
        df ['diag_3'][index]=5
    elif df ['diag_3'][index]>=320 and df ['diag_3'][index]<390:
        df ['diag_3'][index]=6
    elif df ['diag_3'][index]>=390 and df ['diag_3'][index]<460:
        df ['diag_3'][index]=7
    elif df ['diag_3'][index]>=460 and df ['diag_3'][index]<520:
        df ['diag_3'][index]=8
    elif df ['diag_3'][index]>=520 and df ['diag_3'][index]<580:
        df ['diag_3'][index]=9
    elif df ['diag_3'][index]>=580 and df ['diag_3'][index]<630:
        df ['diag_3'][index]=10
    elif df ['diag_3'][index]>=630 and df ['diag_3'][index]<680:
        df ['diag_3'][index]=11
    elif df ['diag_3'][index]>=680 and df ['diag_3'][index]<710:
        df ['diag_3'][index]=12
    elif df ['diag_3'][index]>=710 and df ['diag_3'][index]<740:
        df ['diag_3'][index]=13
    elif df ['diag_3'][index]>=740 and df ['diag_3'][index]<760:
        df ['diag_3'][index]=14
    elif df ['diag_3'][index]>=760 and df ['diag_3'][index]<780:
        df ['diag_3'][index]=15
    elif df ['diag_3'][index]>=780 and df ['diag_3'][index]<800:
        df ['diag_3'][index]=16
    elif df ['diag_3'][index]>=800 and df ['diag_3'][index]<1000:
        df ['diag_3'][index]=17


# In[70]:


df


# In[71]:


df.describe()


# In[72]:


df.drop(['acetohexamide','tolbutamide','troglitazone','examide','citoglipton','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone'],axis=1,inplace=True)


# In[73]:


df[df.duplicated()]


# In[74]:


df.drop_duplicates(inplace=True)


# ## Matplotlib

# In[75]:


plt.hist(df['admission_typeid'])
plt.show()


# In[76]:


plt.hist(df['discharge_disposition_id'])
plt.show()


# In[77]:


plt.hist(df['admission_source_id'])
plt.show()


# In[78]:


plt.hist(df['diag_1'])
plt.show()


# # Training & Testing 

# In[79]:


from sklearn.model_selection import train_test_split


# In[80]:


X = df.drop('label',axis=1)


# In[81]:


y=df['label']


# In[82]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=30)


# In[83]:


print('Shape of X_train : ',X_train.shape)
print('Shape of y_train : ',y_train.shape)


# In[84]:


print('Shape of X_test : ',X_test.shape)
print('Shape of y_test : ',y_test.shape)


# In[ ]:





# Using StandardScaler()

# In[85]:


from sklearn.preprocessing import StandardScaler


# In[86]:


sc=StandardScaler()
sc.fit(X_train)
X_train_sc=sc.transform(X_train)
X_test_sc=sc.transform(X_test)


# ## PCA

# In[87]:


df


# In[88]:


from sklearn.decomposition import PCA

pca=PCA(n_components=34)
X_trainn=pca.fit_transform(X_train_sc)
X_testt=pca.transform(X_test_sc)


# Max When PCA = 34

# # Decision Tree Classifier:

# In[89]:


from sklearn import tree
clf1=tree.DecisionTreeClassifier()


# In[90]:


clf1=clf1.fit(X_train_sc,y_train)
pred1= clf1.predict(X_test_sc)
print(pred1)
clf1.score(X_test_sc,y_test)*100


# In[91]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print ("Accuracy : " , accuracy_score(y_test,pred1)*100)  
print("Report : \n", classification_report(y_test, pred1))
print("F1 Score : ",f1_score(y_test, pred1, average='macro')*100)


# # RandomForest Classifier:

# In[92]:


from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier()
clf2.fit(X_train_sc, y_train)


# In[93]:


clf2=clf2.fit(X_train_sc,y_train)
pred2 = clf2.predict(X_test_sc)
print(pred2)
clf2.score(X_test_sc,y_test)*100


# In[94]:


print ("Accuracy : " , accuracy_score(y_test,pred2)*100)  
print("Report : \n", classification_report(y_test, pred2))
print("F1 Score : ",f1_score(y_test, pred2, average='macro')*100)


# 

# # AdaBoost Classifier:

# In[95]:


from sklearn.ensemble import AdaBoostClassifier


# In[96]:


clf3=AdaBoostClassifier()
clf3.fit(X_train_sc,y_train)
pred3=clf3.predict(X_test_sc)
clf3.score(X_test_sc,y_test)*100


# In[97]:


print ("Accuracy : " , accuracy_score(y_test,pred3)*100)  
print("Report : \n", classification_report(y_test, pred3))
print("F1 Score : ",f1_score(y_test, pred3, average='macro')*100)


# ## Max Accuracy(AdaBoost): 63.22
# ## Min Accuracy (Decision Tree): 55.82

# # Majority Voting Classifier:

# In[98]:


from sklearn.ensemble import VotingClassifier


# In[99]:


eclf1 = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2), ('ab', clf3)], voting='hard')
eclf1 = eclf1.fit(X_train_sc, y_train)
pred4=eclf1.predict(X_test_sc)


# In[100]:


eclf1.score(X_test_sc,y_test)


# In[101]:


print ("Accuracy : " , accuracy_score(y_test,pred4)*100)  
print("Report : \n", classification_report(y_test, pred4))
print("F1 Score : ",f1_score(y_test, pred4, average='macro')*100)


# In[ ]:




