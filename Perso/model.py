import tensorflow as tf
import pandas as pd
import random

from zipfile import ZipFile

random.seed(42)

train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")

def addextension(nm):
    return nm+".jpg"

train_df["Image_ID"] = train_df["Image_ID"].apply(addextension)
test_df["Image_ID"] = test_df["Image_ID"].apply(addextension)

print(train_df.head())

def unzip(nm):
    with ZipFile(nm,"r") as zip:
        zip.extractall()

unzip("Train_Images.zip")
unzip("Test_Images.zip")


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
    horizontal_flip = True,
    rotation_range = 10,
    zoom_range = 0.1,
    validation_split = 0.2,
)

train_ds = train_gen.flow_from_dataframe(
    directory = "Train_Images",
    dataframe = train_df,
    x_col = "Image_ID",
    y_col = "class",
    target_size = (256,256),
    batch_size = 32,
    class_mode = "categorical",
    shuffle = True,
    subset = "training",
)

val_ds = train_gen.flow_from_dataframe(
    directory = "Train_Images",
    dataframe = train_df,
    x_col = "Image_ID",
    y_col = "class",
    target_size = (256,256),
    batch_size = 32,
    class_mode = "categorical",
    shuffle = True,
    subset = "validation",
)

from tensorflow.keras import layers
from tensorflow.keras import Model, Input

def model(input):
    x = layers.Rescaling(1./255)(input)
    x = layers.Conv2D(64,3,activation="relu",padding="same",strides=(2,2))(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(128,3,activation="relu",padding="same",strides=(2,2))(x)
    x = layers.Conv2D(128,3,activation="relu",padding="same",strides=(2,2))(x)
    x = layers.Conv2D(256,3,activation="relu",padding="same",strides=(2,2))(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512,activation="relu")(x)
    x = layers.Dropout(0.2,seed=42)(x)
    x = layers.Dense(512,activation="relu")(x)
    x = layers.Dropout(0.2,seed=42)(x)
    output = layers.Dense(3,activation="softmax")(x)
    model = Model(input,output)
    return model

model = model(Input(shape=(256,256,3)))
model.summary()
model.compile(tf.keras.optimizers.RMSprop(),tf.keras.losses.CategoricalCrossentropy(),metrics=["accuracy"])

if __name__=="__main__":
    checkpoint = tf.keras.callbacks.ModelCheckpoint("makerere.h5",save_weights_only=False,save_best_only=True,monitor="val_accuracy")
    model.fit(train_ds,epochs=20,validation_data=val_ds,callbacks=[checkpoint])
    best = tf.keras.models.load_model("makerere.h5")
    val_loss,val_acc = best.evaluate(val_ds)
    print("\nVal Accuracy: {:.2f} %".format(100*val_acc))
    print("Val Loss: {:.2f} %".format(100*val_loss))