#!/usr/bin/env python
# coding: utf-8

# # Credict Card fraud detection Machine Learning project
# 

# ## Objective
# 1.The main objective of this project is to detect the how many fraud transactions are made by the customers.
# and
# 
# 2.To choose the best model which gives the better accuaracy.

# ## Problem statement
# In this project I am going to predict the fraud caused by the customers. 
# As this is a  classification problem we will be using Logistic Regression, knn, SVM, naive bayes, Random forest, decision models for model training and for predictions.

# #### Imported various libraries like
# 
# Numpy: numpy module is used to do the mathematical calculations
# 
# pandas:pandas is a software library written for the Python programming language for data manipulation and analysis.
# 
# seaborn:Seaborn is used for visualizations of data.
# 
# matplotlib: Matplotlib is used for ploting graphs and charts.
# 
# traintestsplit: Train test split is usd to split the data into training and testing part.
# 
# logistic regression, KNN,SVM, naive bayes, DecisionTree, RandomForestClassifier: This are algorithms which is used to bulit models.
# 
# accuracy Score, classification report, confusion matrix: This is used for model evaluation or to check the performance of the model.

# ## Variables information.
# Imported dataset of credit card fraud detection, the variables in the dataset are:
# 
# Time, V1-V28,Amount,Class
# 
# Time: Time defines in how much time transaction is done.
# 
# V1-V28 : this is confidential data
# 
# Amount: How much amount is transacted
# 
# Class: Tells that the transaction is normal or fraud.

# ## importing important libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score,recall_score,f1_score


# ## importing dataset

# In[2]:


data = pd.read_csv("C:/Users/Gayatri/Downloads/archive (8)/creditcard.csv")


# ## data preprocessing

# In[3]:


data.head()


# In[4]:


data[data["Class"] == 1].tail()


# In[5]:


data["Class"].value_counts()


# In[6]:


data.isna().sum()


# This observation tells that there is no  null values present in the dataset. The data is clean to perform the rest of the operations.

# In[7]:


data.describe()


# In[8]:


data.groupby("Class").mean()


# By using describe method we get the count,mean,standard,minimum value interquartile ranges from 25 to 75 and maximum value of the each and every attribute.

# In[9]:


data.info()


# By using info() method we can see that all of the data variables have same  data type which is float64 but class is of integer data type.

# ### 0 - Normal transaction
# 
# ### 1 - fraud transaction

# In[10]:


real = data[data["Class"] == 0]
fraud = data[data["Class"] == 1]


# In[11]:


print(real.shape)
print(fraud.shape)


# In[12]:


real.Amount.describe()


# In[13]:


fraud.Amount.describe()


# ### Under sampling
# Build a sample dataset containing similar distribution of normal and fraud trnsaction
# 
# Number of fraud transaction -- 492

# Concatenating two dataframe

# In[16]:


from sklearn.utils import resample
real_sample = real.sample(n = 492)
# real_sample = 
new_data = pd.concat([real_sample,fraud],axis = 0)


# In[23]:


# new_data.head()


# In[24]:


# new_data.tail()


# In[17]:


new_data["Class"].value_counts()


# In[18]:


new_data.groupby("Class").mean()


# ## EDA

# In[24]:


corr = data.corr()
corr


# This is the visualization of how the data variables are correlated to each other

# In[21]:


sns.heatmap(corr)


# In[20]:


plt.figure(figsize=(9,9))
plt.subplot(2,1,1)
sns.distplot(new_data["Time"])
plt.subplot(2,1,2)
sns.distplot(new_data["Amount"])


# In[28]:


sns.boxplot(x= new_data["Amount"])


# In[26]:


#find iqr
percentile25 = new_data["Amount"].quantile(0.25)
percentile75 = new_data["Amount"].quantile(0.75)


# In[27]:


iqr=percentile75 - percentile25
iqr


# In[28]:


up=percentile25+1.5*iqr
lr=percentile75-1.5*iqr
print(up)
print(lr)


# In[19]:


# # import library
# from imblearn.over_sampling import SMOTE

# smote = SMOTE()

# # fit predictor and target variable
# x_smote, y_smote = smote.fit_resample(x, y)
 
# print('Original dataset shape', Counter(y))
# print('Resample dataset shape', Counter(y_ros))


# In[29]:


new_data[new_data["Amount"]>up]


# ### Trimming

# In[30]:


newdf = new_data[new_data["Amount"] < up]
newdf.shape


# In[44]:


plt.figure(figsize=(9,9))
plt.subplot(2,2,1)
sns.distplot(new_data["Amount"])
plt.subplot(2,2,2)
sns.boxplot(x = new_data["Amount"])
plt.subplot(2,2,3)
sns.distplot(newdf["Amount"])
plt.subplot(2,2,4)
sns.boxplot(x = newdf["Amount"])
plt.show()


# In[10]:


plt.subplot(2,1,1)
plt.scatter(x = real.Time,y =  real.Amount)
plt.subplot(2,1,2)
plt.scatter(x = fraud.Time, y = fraud.Amount)


# In[70]:


# Get value counts for stroke column
counts = data['Class'].value_counts()

# Set custom labels for pie chart
labels = ['Normal', 'Fraud']

# Create pie chart with custom colors and labels
colors = ['orange', 'red']
plt.pie(counts, colors=colors, labels=labels, autopct='%1.1f%%', startangle=30)

# Add plot title
plt.title("Value count distribution of Class attribute")

# Show plot
plt.show()


# This visualization of Class attribute tells that there only 0.2 percent of fraud transactions are made and rest of the data is normal transaction before the sampling the data.

# #### After balancing the Class attribute we get following result

# In[71]:


# Get value counts for stroke column
counts = new_data['Class'].value_counts()

# Set custom labels for pie chart
labels = ['Normal', 'Fraud']

# Create pie chart with custom colors and labels
colors = ['orange', 'red']
plt.pie(counts, colors=colors, labels=labels, autopct='%1.1f%%', startangle=30)

# Add plot title
plt.title("Value count distribution of Class attribute")

# Show plot
plt.show()


# This visualzation tells that the data is equally distributed after sampling the class attribute.

# In[24]:


plt.figure()
sns.pairplot(new_data, vars=['Time', 'Amount', 'Class'], markers=["o", "s"])
plt.show()


# In the observation of pairplot we can say that the graph between amount and time the transaction of amount is very less or the amount is very less. 
# 
# In between class and time we can say that In a given time transaction of normal and fraud was happened.

# In[25]:


plt.figure(figsize=(9,9))
plt.subplot(2,1,1)
time = new_data["Time"]
amt = new_data["Amount"]
sns.kdeplot(data=new_data, x = "Time", hue ="Class")
plt.subplot(2,1,2)
sns.kdeplot(data=new_data, x = "Amount",hue="Class")


# In[92]:


plt.subplot(2,2,1)
new_data["Time"].hist()
plt.subplot(2,2,2)
new_data["Amount"].hist()
plt.subplot(2,2,3)
new_data["Class"].hist()
plt.show()


# In[27]:


sns.countplot(x ='Class', data = new_data)


# In this count plot we can observe that the data of Class are equally distributed.

# In[62]:


# sns.heatmap(new_data)


# ## divide the data into dependent and independent variable

# ### seperating the dependant and independent variable
# Here the independent variable is Time v1-v28, Amount and droped the column Class because this is the target variable

# In[27]:


x = new_data.drop(columns = "Class", axis = 1)
x


# Get the target variable as Class column which is in categorical type 
# where 0 - Normal transactions
# 1- Fraud transaction

# In[28]:


from sklearn.preprocessing import StandardScaler
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to your data and transform the data
X = scaler.fit_transform(x)


# In Machine Learning, StandardScaler is used to resize the distribution of values so that the mean of the observed values is 0 and the standard deviation is 1.

# In[29]:


#dependent variable
y = new_data["Class"]
y


# ## LogisticRegression model 1

# # train test split
# Train test split is used to sperate the training and testing dataset.
# 
# Where training data is used to fit the model and testing data is used to prediction.

# In[75]:


#splitting the dependent and independent variables into training and testing set
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 2)


# In[76]:


print(xtrain.shape,xtest.shape,ytrain.shape, ytest.shape)


# In[77]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()


# In[78]:


LR.fit(xtrain,ytrain)


# ### model evaluation

# In[79]:


ypred = LR.predict(xtest)
testacc = accuracy_score(ypred, ytest)

print("Model Accuracy:", round(accuracy_score(ytest, ypred),2))
print("Model Precision:", round(precision_score(ytest, ypred),2))
print("Model Recall:", round(recall_score(ytest, ypred),2))
print("Model F1-Score:", round(f1_score(ytest, ypred),2) , '\n')
conf_matrix1 = confusion_matrix(ytest, ypred)
plt.figure(figsize=(6, 6)) 
labels= ['Valid', 'Fraud'] 

sns.heatmap(pd.DataFrame(conf_matrix1),annot=True, fmt='d',
            linewidths= 0.05 ,cmap='BuPu',xticklabels= labels, yticklabels= labels)

print(classification_report(ytest, ypred, target_names=labels) , '\n')

plt.title('LR- Confusion Matrix')
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()


# Logistic Regression model is statical model where evaluations are formed of the
# connection among dependent qualitative variable (binary or binomial logistic regression)
# or variable with three values or higher (multinomial logistic regression) and one
# independent explanatory variable or higher whether qualitative or quantitative.
# The last model created using both python is Logistic Regression, the model
# managed to score and accuracy of 95% in python.
# with 10 misclassified instances.

# ## Random  forest classifier: model2
#  Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset.
# It is a supervised learning algorithm
# It is used for both classification and regression tasks.

# In[39]:


#splitting the dependent and independent variables into training and testing set
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 2)


# In[40]:


from sklearn.ensemble import RandomForestClassifier


# In[41]:


RF = RandomForestClassifier()
RF.fit(xtrain, ytrain)


# In[42]:


ypred1 = RF.predict(xtest)


# In[80]:


print("Model Accuracy:", round(accuracy_score(ytest, ypred1),2)*100)
print("Model Precision:", round(precision_score(ytest, ypred1),2))
print("Model Recall:", round(recall_score(ytest, ypred1),2))
print("Model F1-Score:", round(f1_score(ytest, ypred1),2) , '\n')
conf_matrix2 = confusion_matrix(ytest, ypred1)
plt.figure(figsize=(6, 6)) 
labels= ['Valid', 'Fraud'] 

sns.heatmap(pd.DataFrame(conf_matrix2),annot=True, fmt='d',
            linewidths= 0.05 ,cmap='BuPu',xticklabels= labels, yticklabels= labels)

print(classification_report(ytest, ypred1, target_names=labels) , '\n')

plt.title('RandomForest - Confusion Matrix')
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()


# Random Forest Classifier model is supervised machine learning model which is mostly used for classification.
# This model created using python is Random forest classifier, the model
# managed to score and accuracy of 95% using python with 10 misclassified instances.

# ## DecisionTreeClassifier: model3
# A decision tree is a non-parametric supervised learning algorithm and it is non parametric.
# 
# Used for both classification and regression tasks.
# 
# It has a hierarchical, tree structure, which consists of a root node, branches, internal nodes and leaf nodes.
# 

# In[44]:


#splitting the dependent and independent variables into training and testing set
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 2)


# In[45]:


from sklearn.tree import DecisionTreeClassifier


# In[46]:


DT = DecisionTreeClassifier(criterion="entropy")


# In[47]:


DT.fit(xtrain,ytrain)


# In[48]:


ypred2 = DT.predict(xtest)


# In[50]:


print("Model Accuracy:", round(accuracy_score(ytest, ypred2),2)*100)
print("Model Precision:", round(precision_score(ytest, ypred2),2))
print("Model Recall:", round(recall_score(ytest, ypred2),2))
print("Model F1-Score:", round(f1_score(ytest, ypred2),2) , '\n')
conf_matrix1 = confusion_matrix(ytest, ypred2)
plt.figure(figsize=(6, 6)) 
labels= ['Valid', 'Fraud'] 

sns.heatmap(pd.DataFrame(conf_matrix1),annot=True, fmt='d',
            linewidths= 0.05 ,cmap='BuPu',xticklabels= labels, yticklabels= labels)

print(classification_report(ytest, ypred2, target_names=labels) , '\n')

plt.title('Decision tree classifier - Confusion Matrix')
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()


# This is model of decision tree which is used to perform classification and regression task and in this case classification task is performed observed that the accuracy is 89% and 21 misclassified instances.

# ## Kth nearest neighobur model 4

# In[51]:


#splitting the dependent and independent variables into training and testing set
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 2)


# In[52]:


#Fitting K-NN classifier to the training set  
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)  
classifier.fit(xtrain, ytrain)  


# In[53]:


ypred5 = classifier.predict(xtest)


# In[54]:


print("Model Accuracy:", round(accuracy_score(ytest, ypred5),2)*100)
print("Model Precision:", round(precision_score(ytest, ypred5),2))
print("Model Recall:", round(recall_score(ytest, ypred5),2))
print("Model F1-Score:", round(f1_score(ytest, ypred5),2) , '\n')
conf_matrix4 = confusion_matrix(ytest, ypred5)
plt.figure(figsize=(6, 6)) 
labels= ['Valid', 'Fraud'] 

sns.heatmap(pd.DataFrame(conf_matrix4),annot=True, fmt='d',
            linewidths= 0.05 ,cmap='BuPu',xticklabels= labels, yticklabels= labels)

print(classification_report(ytest, ypred5, target_names=labels) , '\n')

plt.title('KNN - Confusion Matrix')
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()


# In this model of KNN we can see that the accuracy is 93% with 13 missclassified instances.

# ## support vector machine model 5

# In[55]:


#splitting the dependent and independent variables into training and testing set
# xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 2)


# In[30]:


from sklearn.svm import SVC # "Support vector classifier"  
classifier2 = SVC(kernel='linear', random_state=42)  
classifier2.fit(x, y) 


# In[35]:


# w = classifier2.coef_[0]
# b = classifier2.intercept_[0]
# slope = -w[0]/w[1]
# inter = -b/w[1]


# In[1]:


# plt.scatter(x[:,0],x[:,1],c =y,cmap = "viridis",edgecolors = "k",s  =30)


# In[57]:


ypred6 = classifier2.predict(xtest)


# In[81]:


print("Model Accuracy:", round(accuracy_score(ytest, ypred6),2)*100)
print("Model Precision:", round(precision_score(ytest, ypred6),2))
print("Model Recall:", round(recall_score(ytest, ypred6),2))
print("Model F1-Score:", round(f1_score(ytest, ypred6),2) , '\n')
conf_matrix1 = confusion_matrix(ytest, ypred6)
plt.figure(figsize=(6, 6)) 
labels= ['Valid', 'Fraud'] 

sns.heatmap(pd.DataFrame(conf_matrix1),annot=True, fmt='d',
            linewidths= 0.05 ,cmap='BuPu',xticklabels= labels, yticklabels= labels)

print(classification_report(ytest, ypred6, target_names=labels) , '\n')

plt.title('SVM- Confusion Matrix')
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()


# ### observation
# Support Vector machine is a supervised ML technique with connected learning algorithms
# which inspect data used for both classification and regression analyses, it also performs
# linear classification, additionally to non-linear classification by creating margins between
# the classes, which are created in such a fashion that the space between the margin and
# the classes is maximum which minimizes the error of the classification.
# Finally, the model Support Vector Machine managed to score
# 95.0% for the accuracy and misclassified 10 instances.

# ## Naive bayes -model 6
# Naive Bayes is a family of probabilistic machine learning algorithms based on Bayes' theorem, with a "naive" assumption of independence among the features.The various classification tasks, including text classification, spam email detection, sentiment analysis.

# In[82]:


#splitting the dependent and independent variables into training and testing set
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 2)


# In[83]:


from sklearn.naive_bayes import GaussianNB  
classifier3 = GaussianNB()  
classifier3.fit(xtrain, ytrain) 


# In[84]:


ypred7 = classifier3.predict(xtest)


# In[85]:


print("Model Accuracy:", round(accuracy_score(ytest, ypred7),2)*100)
print("Model Precision:", round(precision_score(ytest, ypred7),2))
print("Model Recall:", round(recall_score(ytest, ypred7),2))
print("Model F1-Score:", round(f1_score(ytest, ypred7),2) , '\n')
conf_matrix1 = confusion_matrix(ytest, ypred7)
plt.figure(figsize=(6, 6)) 
labels= ['Valid', 'Fraud'] 

sns.heatmap(pd.DataFrame(conf_matrix1),annot=True, fmt='d',
            linewidths= 0.05 ,cmap='BuPu',xticklabels= labels, yticklabels= labels)

print(classification_report(ytest, ypred7, target_names=labels) , '\n')

plt.title('Naive bayes - Confusion Matrix')
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()


# In the result naive bayes used for classification problems and it worked well.In the model evaluation of naive bayes we can observed that we get the accuracy of 94% and 12 misclassified instances.

# ### comparison
# Logistic regression : accuracy- 95 and 10 misclassification
# Random forest : accuracy - 93 and 14 misclassification
# Decisiontree: accuracy - 89 and 21 misclassification
# KNN : accuracy 93% and 13 misclassification
# SVM : accuracy 95% and 10 misclassification
# Naive bayes: accuracy- 91% and 18 misclassification

# ## Conclusion
# In conclusion, the main objective of this project was to find the most suited model in credit
# card fraud detection in terms of the machine learning techniques chosen for the project,
# and it was met by building the  models and finding the accuracies of them all, the best
# model in terms of accuracies is LogisticRegression and svm which scored 95% with only
# 10 misclassified instances. I believe that using the model will help in decreasing the
# amount of credit card fraud and increase the customers satisfaction as it will provide them
# with better experience in addition to feeling secure.

# In[ ]:




