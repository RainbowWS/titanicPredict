
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# In[26]:


#读取文件
data_train = pd.read_csv("F:\\aliTIANCHILearn\\titanic_train.csv")
data_test = pd.read_csv("F:\\aliTIANCHILearn\\titanic_test.csv")


# In[27]:


data_train.head()


# In[28]:


data_train.info()


# In[29]:


data_train.describe()


# In[30]:


#由上可知Age存在null值，对Age列中的缺失值用Age中位数进行填充
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())

#对于数据null值或者缺失值可以使用以下方法处理：
#1.使用可用特征的均值来填补缺失值
#2.特殊值来填补缺失值，例如：-1、-2等
#3.直接忽略有缺失值的样本
#4.使用有相似样本的平均值添补缺失值
#5.使用另外的机器学习算法来预测缺失值


# In[31]:


data_train.describe()


# In[32]:


#线性回归
from sklearn.linear_model import LinearRegression   

#训练集交叉验证，得到平均值
from sklearn.model_selection import KFold
 
#选取简单的可用输入特征
predictors = ["Pclass","Age","SibSp","Parch","Fare"]     
#predictors = ["Age"]    
 
#初始化线性回归算法
alg = LinearRegression()
#样本平均分成3份，3折交叉验证
#kf = KFold(data_train.shape[0],n_folds=3,random_state=1)   
kf = KFold(n_splits=3,shuffle=False,random_state=1) 

predictions = []
for train,test in kf.split(data_train):
    train_predictors = (data_train[predictors].iloc[train,:])
    train_target = data_train["Survived"].iloc[train]
    alg.fit(train_predictors,train_target)
    test_predictions = alg.predict(data_train[predictors].iloc[test,:])
    predictions.append(test_predictions)


# In[33]:


import numpy as np
 
#The predictions are in three aeparate numpy arrays.	Concatenate them into one.
#We concatenate them on axis 0,as they only have one axis.
predictions = np.concatenate(predictions,axis=0)
 
#Map predictions to outcomes(only possible outcomes are 1 and 0)
predictions[predictions>.5] = 1
predictions[predictions<=.5] = 0
accuracy = sum(predictions == data_train["Survived"]) / len(predictions)
print ("准确率为: ", accuracy)


# In[34]:


#逻辑回归算法

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

#初始化逻辑回归
LogRegAlg = LogisticRegression(random_state=1)
re = LogRegAlg.fit(data_train[predictors],data_train["Survived"])

#使用sklearn库里面的交叉验证函数获取预测准确分数
scores = model_selection.cross_val_score(LogRegAlg, data_train[predictors], data_train["Survived"],cv=3)

print("准确率为:",scores.mean())


# In[35]:


#增加特征Sex和EmbarEmbarked
data_train.head()


# In[36]:


#对sex进行处理，如果是male用0代替，是female用1代替
data_train.loc[data_train["Sex"] == "male","Sex"] = 0
data_train.loc[data_train["Sex"] == "female","Sex"] = 1


# In[37]:


#查看发现Embarked列出现最多为S，缺失值用最多的S进行填充
data_train["Embarked"] = data_train["Embarked"].fillna('S') 
#地点用0,1,2
data_train.loc[data_train["Embarked"] == "S","Embarked"] = 0    
data_train.loc[data_train["Embarked"] == "C","Embarked"] = 1
data_train.loc[data_train["Embarked"] == "Q","Embarked"] = 2


# In[38]:


#增加了Sex和Embarked两个特征，继续用逻辑回归进行预测
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]  

LogRegAlg=LogisticRegression(random_state=1)
#Compute the accuracy score for all the cross validation folds.(much simpler than what we did before!)
re = LogRegAlg.fit(data_train[predictors],data_train["Survived"])
scores = model_selection.cross_val_score(LogRegAlg,data_train[predictors],data_train["Survived"],cv=3)
#Take the mean of the scores (because we have one for each fold)
print("准确率为: ",scores.mean())


# In[39]:


#通过增加了特征，发现准确率提高，说明好的特征的提取有利于提高预测准确度


# In[40]:


data_test.describe()


# In[41]:


#使用测试集数据进行预测。
#新增：对测试集数据进行预处理，并进行结果预测
#Age列中的缺失值用Age均值进行填充
data_test["Age"] = data_test["Age"].fillna(data_test["Age"].median())
#Fare列中的缺失值用Fare最大值进行填充
data_test["Fare"] = data_test["Fare"].fillna(data_test["Fare"].max()) 

#Sex性别列处理：male用0，female用1
data_test.loc[data_test["Sex"] == "male","Sex"] = 0
data_test.loc[data_test["Sex"] == "female","Sex"] = 1
#缺失值用最多的S进行填充
data_test["Embarked"] = data_test["Embarked"].fillna('S') 
#地点用0,1,2
data_test.loc[data_test["Embarked"] == "S","Embarked"] = 0    
data_test.loc[data_test["Embarked"] == "C","Embarked"] = 1
data_test.loc[data_test["Embarked"] == "Q","Embarked"] = 2

test_features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"] 
#先生成测试集的Survived，并默认值为-1
data_test["Survived"] = -1
#输入特征
test_predictors = data_test[test_features]
#得出预测结果----Survived列
data_test["Survived"] = LogRegAlg.predict(test_predictors)


# In[42]:


#查看结果
data_test.head(10)


# In[ ]:




