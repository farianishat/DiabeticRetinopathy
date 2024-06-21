#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Install numpy
# !pip install numpy

# # Install pandas
# !pip install pandas

# # Install matplotlib
# !pip install matplotlib

# # Install seaborn
# !pip install seaborn

# # Install TensorFlow
# !pip install tensorflow

# # Install scikit-learn
# !pip install scikit-learn


# In[8]:


# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


# In[12]:


# Load and Inspect the Data
csv_file = r'C:\Users\nisha\Desktop\Projects\DiabeticRetinopathy\Dataset\messidor_data.csv'  # CSV file path
df = pd.read_csv(csv_file)


# In[13]:


# Display the first few rows of the dataset
print(df.head())


# In[14]:


# Filter out ungradable images
df = df[df['adjudicated_gradable'] == 1]

# Drop the 'adjudicated_gradable' column as it's not needed anymore
df = df.drop(columns=['adjudicated_gradable'])


# In[19]:


# Convert the 'adjudicated_dr_grade' column to string
df['adjudicated_dr_grade'] = df['adjudicated_dr_grade'].astype(str)


# In[20]:


# Display the class distribution of DR and DME
print("Distribution of DR grades:")
print(df['adjudicated_dr_grade'].value_counts())
print("Distribution of DME grades:")
print(df['adjudicated_dme'].value_counts())


# In[28]:


# Determine the number of unique classes for the target variable
num_classes = df['adjudicated_dr_grade'].nunique()
print(f"Number of classes: {num_classes}")


# In[21]:


# Prepare the image data generator
image_dir = r'C:\Users\nisha\Desktop\Projects\DiabeticRetinopathy\Images'  # Update this to your actual image directory path
img_height, img_width = 224, 224
batch_size = 32


# In[22]:


# Data generator for training and validation
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Split 20% of data for validation
)


# In[23]:


# Data generator for training
train_generator = data_gen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='image_id',
    y_col='adjudicated_dr_grade',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)


# In[24]:


# Data generator for validation
validation_generator = data_gen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='image_id',
    y_col='adjudicated_dr_grade',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# In[25]:


# Build the Model Using Transfer Learning
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))


# In[26]:


# Freeze the base model
base_model.trainable = False


# In[29]:


# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)


# In[30]:


# Define the model
model = Model(inputs=base_model.input, outputs=predictions)


# In[31]:


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[32]:


# Train the Model
epochs = 10

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)


# In[33]:


# Evaluate the Model
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")


# In[34]:


# Confusion Matrix and Classification Report
validation_generator.reset()
preds = model.predict(validation_generator, steps=validation_generator.samples // batch_size + 1)
y_pred = np.argmax(preds, axis=1)
y_true = validation_generator.classes[:len(y_pred)]


# In[35]:


print("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
print(cm)


# In[36]:


print("Classification Report")
cr = classification_report(y_true, y_pred, target_names=[str(i) for i in range(num_classes)])
print(cr)


# In[40]:


def plot_history(history):
    # Plot accuracy and loss over epochs
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax[0].plot(history.history['accuracy'], label='train accuracy')
    ax[0].plot(history.history['val_accuracy'], label='validation accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(loc='best')
    
    # Plot training & validation loss
    ax[1].plot(history.history['loss'], label='train loss')
    ax[1].plot(history.history['val_loss'], label='validation loss')
    ax[1].set_title('Model Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(loc='best')
    
    plt.tight_layout()
    plt.show()

plot_history(history)

