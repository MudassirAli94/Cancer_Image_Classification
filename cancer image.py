
# coding: utf-8

# In[1]:


# General Packages
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time

# General Mathematics package
import math as math

# Graphing Packages
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")

# Statistics Packages
from scipy.stats import randint
from scipy.stats import skew

# Machine Learning Packages
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale
from sklearn import preprocessing
from skimage.transform import resize
import xgboost as xgb

# Neural Network Packages
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf

# H2o packages
import h2o
from h2o.automl import H2OAutoML


# In[2]:


cancer = pd.read_csv("data.csv")
cancer.head()


# In[3]:


pixel_8 = pd.read_csv("hmnist_8_8_L.csv")
pixel_8_rgb = pd.read_csv("hmnist_8_8_RGB.csv")


# In[4]:


cancer.shape


# In[5]:


cancer.localization.value_counts().plot.bar()
plt.show()

# From the graph we can see that on the human body, the back had the most tumor cells
# In[6]:


cancer.dx_type.value_counts().plot.bar()
plt.show()


# In[7]:


cancer.dx.value_counts().plot.bar()
plt.show()

# We can see nv is the most common cancer type in the data set
# In[8]:


pd.crosstab(cancer.age , cancer.sex).drop("unknown" , axis = 1).plot.bar()
plt.show()

# The most common age for both men and women for cancer is 45
# In[9]:


pd.crosstab(cancer.dx , cancer.sex).drop("unknown" , axis = 1).plot.bar()
plt.show()

# The most common type of cancer for both men and women is nv
# In[10]:


train = cancer.sort_values('image_id')[0:5000]
test = cancer.sort_values('image_id')[5000:]


# In[11]:


le = preprocessing.LabelEncoder()

y_train = le.fit_transform(train.dx)
y_test = le.fit_transform(test.dx)


# In[12]:


cancer.iloc[4349]


# In[13]:


from skimage import io


# In[14]:


imgbt = io.imread('cancer image 1/ISIC_0024306.jpg')


# In[15]:


imgbt.shape

# One way to graph the images is to use the glob functionimport globimages = sorted(glob.glob("cancer image 1/*.jpg"))imgbl = []
# for i in images:
   # imgbl.append(io.imread(i))images = pd.Series(imgbl)train['images'] = imagestrain.head()

# Another way is to use a for loop with the append function on an empty list
# In[16]:


pixel_train = []
for n in range(0,len(train)):
    pixel_train.append(io.imread('cancer image 1/' + str(train.image_id.iloc[n]) + '.jpg'))
    
pixel_test = []
for n in range(0,len(test)):
    pixel_test.append(io.imread('cancer image 2/' + str(test.image_id.iloc[n]) + '.jpg'))


# In[17]:


images = pd.Series(pixel_train)
images_test = pd.Series(pixel_test)


# In[18]:


train["images"] = images
test["images"] = images_test
train.head()


# In[19]:


plt.imshow(train.images[0])
plt.show()

# Now that we concated the array of all the images on the train and test data we can continue with our neural network models.
# In[20]:


pixel_train = np.array(pd.Series(pixel_train))
pixel_test = np.array(pd.Series(pixel_test))


# In[21]:


warnings.filterwarnings("ignore")

rs = []

for n in range(0,len(train)):
    rs.append(resize(pixel_train[n] , (75,100)))
    
rs_test = []

for n in range(0, len(test)):
    rs_test.append(resize(pixel_test[n] , (75,100)))


# In[22]:


rs[0].shape


# In[23]:


train["resized images"] = rs
test["resized images"] = rs_test


# In[24]:


X_train = rs
X_test = rs_test


# In[25]:


X_train = np.array(X_train)
X_test = np.array(X_test)


# In[26]:


y_train = np_utils.to_categorical(y_train, 7)
y_test = np_utils.to_categorical(y_test , 7)

# Using Keras on the proportionally split data
# In[27]:


early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

model = Sequential()
model.add(Conv2D(32,(3,3) , input_shape = (75,100,3) , activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
for n in range(1,200):
    model.add(Dense(units = n , activation = 'relu'))
model.add(Dense(units = 7 , activation = 'sigmoid'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])


# In[28]:


model.fit(X_train , y_train, batch_size = 30 ,  epochs = 3, validation_data = (X_test , y_test) , validation_split=0.2,
          verbose = 2 , callbacks=[early_stopping_monitor])

# Using TensorFlow on our proportionally split data
# In[29]:


early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

for n in range (1,200):
    model_tf = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32,(3,3) , input_shape = (75,100,3) , activation = tf.nn.relu),
      tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(n, activation=tf.nn.relu),
      tf.keras.layers.Dense(7 , activation=tf.nn.sigmoid)
    ])
model_tf.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_tf.fit(X_train , y_train, batch_size = 30 ,  epochs = 20, validation_data = (X_test , y_test) , validation_split=0.2,
          verbose = 2 , callbacks=[early_stopping_monitor])

# As you can see tensorflow gives me the best accuracy at 74% accuracy.
# To increase the accuracy I am going to see if decreasing the train data and increasing the test data will work
# In[30]:


train_1 = cancer.sort_values('image_id')[0:2000]
test_1 = cancer.sort_values('image_id')[2000:]


# In[31]:


pixel_train_1 = []
for n in range(0,len(train_1)):
    pixel_train_1.append(io.imread('cancer image 1/' + str(train_1.image_id.iloc[n]) + '.jpg'))
    
pixel_test_1 = []
for n in range(0,3000):
    pixel_test_1.append(io.imread('cancer image 1/' + str(test_1.image_id.iloc[n]) + '.jpg'))
for n in range(3000,len(test_1)):    
    pixel_test_1.append(io.imread('cancer image 2/' + str(test_1.image_id.iloc[n]) + '.jpg'))


# In[32]:


pixel_train_1 = np.array(pd.Series(pixel_train_1))
pixel_test_1 = np.array(pd.Series(pixel_test_1))


# In[33]:


warnings.filterwarnings("ignore")

rs_1 = []

for n in range(0,len(train_1)):
    rs_1.append(resize(pixel_train_1[n] , (75,100)))
    
rs_test_1 = []

for n in range(0, len(test_1)):
    rs_test_1.append(resize(pixel_test_1[n] , (75,100)))


# In[34]:


train_1["resized images"] = rs_1
test_1["resized images"] = rs_test_1


# In[35]:


X_train_1 = rs_1
X_test_1 = rs_test_1
y_train_1 = le.fit_transform(train_1.dx)
y_test_1 = le.fit_transform(test_1.dx)


# In[36]:


X_train_1 = np.array(X_train_1)
X_test_1 = np.array(X_test_1)
y_train_1 = np_utils.to_categorical(y_train_1, 7)
y_test_1 = np_utils.to_categorical(y_test_1 , 7)


# In[37]:

# Keraas on the unproportional dataset

early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
model = Sequential()
model.add(Conv2D(32,(3,3) , input_shape = (75,100,3) , activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dense(units = 7 , activation = 'sigmoid'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])


# In[38]:


model.fit(X_train_1 , y_train_1, batch_size = 30 ,  epochs = 20, validation_data = (X_test_1 , y_test_1) , validation_split=0.2,
          verbose = 2 , callbacks=[early_stopping_monitor])

# Tensor flow on the unproportional dataset
# In[39]:


early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32,(3,3) , input_shape = (75,100,3) , activation = tf.nn.relu),
  tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(200 , activation = tf.nn.relu),
  tf.keras.layers.Dense(7 , activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_1 , y_train_1, batch_size = 30 ,  epochs = 20, validation_data = (X_test_1 , y_test_1) , validation_split=0.2,
          verbose = 2 , callbacks=[early_stopping_monitor])

# The best accuracy I obtained was via tensorflow when the data was on our unproportional split for train and test data.
# I could tune the parameters of the neural networks however my machine does not have the necessary power to run these tuned up parameters in an appropriate time.