#!/usr/bin/env python
# coding: utf-8

# **This is a notebook for an ML model that can predict breast cancer as either Benign(class 1) or Malignant(class 0)The wisconsin breast cancer data sets was gotten from Kaggle and it contains Features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.***

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle 


# **Upload the data as bc_data from my google colab, bear in mind that you have to manually upload this data to colab(click on files on the top left corner of google colab, select the arrow pointing upwards and click on it, then it will open up youe local directories, just navigate to where your data is saved then select and colab will upload it, alternatively you can save your data on your google drive then mount you drive on google colab**

# In[2]:


bc_data=pd.read_csv('/content/drive/MyDrive/data.csv')


# In[3]:


bc_data.head()


# In[4]:


#lets convert the diagnosis outcome from object type to an integer, where Malignant(M)=0 and Benign(B)=1, this will be our two classes
bc_data['diagnosis']=bc_data['diagnosis'].replace({'M':0,'B':1})


# In[5]:


bc_data


# In[6]:


bc_data = bc_data.rename(columns={"concave points_mean":"concave_points_mean"})
bc_data.concave_points_mean


# In[7]:


#Checking for missing values
bc_data.isna().count()
#It appears there are no missing values.


# **The data already has been cleaned, there are no missing values all the features required for training are numerical and our target vector which is the diagnosis is in Binary form thus good for binary classification
# The code below will be used for feature selection to select our X and Y , the feature matrix and target vector(classes)**

# In[9]:


X=np.asarray(bc_data[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean']])


# In[10]:


X


# In[11]:


y=(bc_data['diagnosis'])


# In[12]:


y


# In[13]:


#Let's plot a count plot for Y to visualize the count of Malignant and Benign diagnosis from the data
sns.countplot(x="diagnosis", data=bc_data)


# In[14]:


y.value_counts()


# In[15]:


#percentage of Benign and malignant diagnosis from total of all diagnosis can be calculated as
percent_B= 357/569*100
percent_M= 212/569*100
print(f'percentage of Benign(B) tumor from all diagnosis of the data is {percent_B} % ')
print(f'percentage of Malignanat(M) tumor from all diagnosis of the data is {percent_M} %')


# Let us normalize our feature matrix, the X variable.

# In[16]:


#Normalization using standard scaler, to make the mean of the feature matrix 0 and standard deviation 1
from sklearn import preprocessing 

StandardScaler=preprocessing.StandardScaler()
#fit transform the feature
X=StandardScaler.fit_transform(X)
X





# Now that our Features and target are known, we can go ahead to split the data into train and test sets

# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=5)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# **Next we train our Classification model**

# In[18]:


from sklearn.neighbors import KNeighborsClassifier
bc_predictor=KNeighborsClassifier(n_neighbors = 2).fit(X_train,y_train) 
bc_predictor


# In[19]:


#make prediction on the test data with the model named bc_predictor
y_pred = bc_predictor.predict(X_test)
y_pred[0:5]


# In[20]:


#use some evaluation metrics to check for the performance of our model
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, bc_predictor.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# In[21]:


#use confusion matrix for evaluation
from sklearn.metrics import classification_report, confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1])

#function plots our confusion matrix 
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['diagnosis=0','diagnosis=1'],normalize= False,  title='Confusion matrix')


# In[22]:


#print the accuracy, precision, recall and f1-score of the model..Rememeber a good model tends to value of one for these metrics.
print (classification_report(y_test, y_pred))


# **We have built our model with an accuracy of 0.88 that is 88% accuracy now lets save it as a file i prepration for MLOPS**

# In[23]:


#Save model using Pickel, you can as well use Joblib, they are two different packages that do same thing
import pickle


# In[24]:


#filename = 'finalized_model.sav'
#pickle.dump(bc_predictor, open(filename, 'wb'))

pickle.dump(bc_predictor, open('model.pkl', 'wb'))


# In[25]:


#load and test the pickled model
pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model.predict(X_test)[0:5]

