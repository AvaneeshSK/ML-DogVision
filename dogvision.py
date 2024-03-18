# Dog Breed Classification -> all types of modelling experiments
# https://www.kaggle.com/competitions/dog-breed-identification

# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import tensorflow as tf
import tensorflow_hub as hub

from keras.applications.efficientnet_v2 import EfficientNetV2B3

from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('Machine Learning 2/dog-breed-identification/labels.csv')
print(df)

# training imgs
train_folder_path = 'C:/Users/aviaf/Documents/Python/Machine Learning 2/dog-breed-identification/train'
train_img_paths = []
ids = list(df['id'])
for id in ids:
    train_img_paths.append(f'{train_folder_path}/{id}.jpg')

# testing imgs
test_folder_path = os.listdir('C:/Users/aviaf/Documents/Python/Machine Learning 2/dog-breed-identification/test')
test_img_paths = []
for each in test_folder_path:
    test_img_paths.append(f'C:/Users/aviaf/Documents/Python/Machine Learning 2/dog-breed-identification/test/{each}')

    
# onehotencode labels in boolean
unique_labels = df['breed'].unique()
labels = list(df['breed']) # actual labels
onehot_labels = [breed == unique_labels for breed in list(df['breed'])]

# onehotencode labels in int boolean
onehot_int_labels = []
for label in onehot_labels:
    temp = []
    for each in label:
        if each == True:
            temp.append(1)
        else:
            temp.append(0)
    onehot_int_labels.append(temp)

# preprocessing 
def preprocessing(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])
    return img

train_preprocessed_imgs = []
for img_path in train_img_paths[:9500]:
    train_preprocessed_imgs.append(preprocessing(img_path))

test_preprocessed_imgs = []
for img_path in test_img_paths[:10]:
    test_preprocessed_imgs.append(preprocessing(img_path))

# training on 8000 images just to see, using 50 images of training images as test images
X = np.array(train_preprocessed_imgs)[:8000] 
y = np.array(onehot_int_labels)[:8000]

X_test = np.array(train_preprocessed_imgs)[8000:9500]
y_test = np.array(onehot_int_labels)[8000:9500]

# creating data batches (if needed for efficiency this can be used)
# def preprocessing_for_batches(img, label): # calling above preprocessing, accomodate labels
#     return preprocessing(img), label

# def create_data_batches(X_train=None, y_train=None, train=False):
#     if train:
#         dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X_train), tf.constant(y_train)))
#         dataset = dataset.map(preprocessing_for_batches)
#         dataset = dataset.shuffle(buffer_size=len(X_train))
#         dataset = dataset.batch(batch_size=32)
#         return dataset
# print(create_data_batches(X_train=train_img_paths[:1000], y_train=y[:1000], train=True))


# modelling 

# 1. ANN Model
ann_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=25),
    tf.keras.layers.Dense(units=25),
    tf.keras.layers.Dense(units=120, activation='softmax')
])

ann_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(), # requires labels to be onehotencoded
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

ann_model.fit(
    x=X,
    y=y,
    verbose=2,
    batch_size=32,
    epochs=25,
    shuffle=True
)

preds = ann_model.predict(X_test)
preds_text = []
for pred in preds:
    preds_text.append(labels[np.argmax(pred)])
actuals = []
for i in range(8000, 9500):
    actuals.append(labels[i])
ann_model_score = accuracy_score(y_pred=preds_text, y_true=actuals)

# 2. CNN Model
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        padding='valid',
        pool_size=2
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        padding='valid',
        pool_size=2
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        padding='valid',
        pool_size=2
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=120, activation='softmax')
])

cnn_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(), # requires labels to be onehotencoded
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

cnn_model.fit(
    x=X,
    y=y,
    verbose=2,
    batch_size=32,
    epochs=25,
    shuffle=True
)

preds = cnn_model.predict(X_test)
preds_text = []
for pred in preds:
    preds_text.append(labels[np.argmax(pred)])
actuals = []
for i in range(8000, 9500):
    actuals.append(labels[i])
cnn_model_score = accuracy_score(y_pred=preds_text, y_true=actuals)

# 3. Pre-trained CNN
efficientnet_v2_url = 'https://www.kaggle.com/models/google/efficientnet-v2/frameworks/TensorFlow2/variations/imagenet1k-b0-classification/versions/2'

cnn_pretrained_model = tf.keras.Sequential([
    hub.KerasLayer(efficientnet_v2_url, trainable=False, input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(units=120, activation='softmax')
])

cnn_pretrained_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

cnn_pretrained_model.fit(
    x=X,
    y=y,
    epochs=2,
    batch_size=32,
    shuffle=True
)

preds = cnn_pretrained_model.predict(X_test)
preds_text = []
for pred in preds:
    preds_text.append(labels[np.argmax(pred)])
actuals = []
for i in range(8000, 9500):
    actuals.append(labels[i])
cnn_pretrained_score = accuracy_score(y_pred=preds_text, y_true=actuals)

# 4. Pre-trained CNN, layers freezed, batch norm stats freezed
base_model = EfficientNetV2B3(include_top=False, include_preprocessing=False)
base_model.trainable = False # freeze layers
input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
x = base_model(input_layer, training=False) # freeze batch norm stats
pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(x)
output_layer = tf.keras.layers.Dense(units=120, activation='softmax')(pooling_layer)
model = tf.keras.Model(input_layer, output_layer)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(
    x=X, 
    y=y,
    verbose=2,
    batch_size=32,
    epochs=2,
    shuffle=True
)

preds = model.predict(X_test)
preds_text = []
for pred in preds:
    preds_text.append(labels[np.argmax(pred)])
actuals = []
for i in range(8000, 9500):
    actuals.append(labels[i])
cnn_pretrained_freezed_score = accuracy_score(y_pred=preds_text, y_true=actuals)

# 5. Pre-trained CNN (Fine Tuning), top 10 layers unfreezed, batch norm stats freezed
base_model = EfficientNetV2B3(include_top=False, include_preprocessing=False)
base_model.trainable=False
input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
x = base_model(input_layer, training=False)
pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(x)
output_layer = tf.keras.layers.Dense(units=120, activation='softmax')(pooling_layer)

model = tf.keras.Model(input_layer, output_layer)

# get the base model layers 
functional_layer = model.layers[1]
for i, layer in enumerate(functional_layer.layers):
    print(f'{i} {layer} can be trained : {layer.trainable}\ n')

# unfreeze top 10 layers
for layer in functional_layer.layers[-10:]:
    layer.trainable = True

# check if they are unfreezed
for i, layer in enumerate(functional_layer.layers):
    print(f'{i} {layer} can be trained : {layer.trainable}\n')

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(
    x=X,
    y=y,
    verbose=2,
    batch_size=32,
    epochs=2
)

preds = model.predict(X_test)
preds_text = []
for pred in preds:
    preds_text.append(labels[np.argmax(pred)])
actuals = []
for i in range(8000, 9500):
    actuals.append(labels[i])
cnn_pretrained_unfreezed_score = accuracy_score(y_pred=preds_text, y_true=actuals)

# plot scores
fig, ax = plt.subplots()
ax.bar(['ANN', 'CNN', 'CNN Prebuilt', 'CNN Feature Extractor', 'CNN Fine Tuning'], [ann_model_score, cnn_model_score, cnn_pretrained_score, cnn_pretrained_freezed_score, cnn_pretrained_unfreezed_score], color=['salmon', 'pink', 'lightblue', 'seagreen', 'wheat'])
ax.set(xlabel='Different Models/Methods', ylabel='Accuracy Score', title='Models/Methods vs Score')
ax.set_xticks(np.arange(0, 5, 1))
ax.set_xticklabels(['ANN', 'CNN', 'CNN Prebuilt', 'CNN Feature Extractor', 'CNN Fine Tuning'], rotation=45)

# heatmap of prediction labels
fig, ax2 = plt.subplots()
sns.heatmap(data=confusion_matrix(y_pred=preds_text, y_true=actuals), annot=True, xticklabels=unique_labels, yticklabels=unique_labels, fmt='.2f', ax=ax2)
plt.show()
