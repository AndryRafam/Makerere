import tensorflow as tf
import numpy as np
import random
import pandas as pd

from zipfile import ZipFile

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")

def addextension(nm):
    return nm+".jpg"

train_df["Image_ID"] = train_df["Image_ID"].apply(addextension)
test_df["Image_ID"] = test_df["Image_ID"].apply(addextension)

print(train_df.head())

with ZipFile("Train_Images.zip","r") as zip:
    zip.extractall()

with ZipFile("Test_Images.zip","r") as zip:
    zip.extractall()


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

rescale = tf.keras.applications.vgg19.preprocess_input
base_model = tf.keras.applications.VGG19(input_shape=(256,256,3),include_top=False,weights="imagenet")
base_model.trainable = True

for layer in base_model.layers:
    if layer.name == "block3_pool":
        break
    layer.trainable = False

class Transfer_VGG19():
    def model(self,input):
        self.x = rescale(input)
        self.x = base_model(self.x,training=False)
        self.x = tf.keras.layers.GlobalAveragePooling2D()(self.x)
        self.x = tf.keras.layers.Flatten()(self.x)
        self.x = tf.keras.layers.Dense(128,activation="relu")(self.x)
        self.x = tf.keras.layers.Dropout(0.2,seed=42)(self.x)
        self.x = tf.keras.layers.Dense(64,activation="relu")(self.x)
        self.x = tf.keras.layers.Dropout(0.2,seed=42)(self.x)
        self.output = tf.keras.layers.Dense(3,activation="softmax")(self.x)
        self.model = tf.keras.Model(input,self.output,name="Transfer_VGG19")
        return self.model

TFVGG19 = Transfer_VGG19()
model = TFVGG19.model(tf.keras.Input(shape=(256,256,3)))
model.summary()
model.compile(tf.keras.optimizers.RMSprop(1e-5),tf.keras.losses.CategoricalCrossentropy(),metrics=["accuracy"])

if __name__=="__main__":
    checkpoint = tf.keras.callbacks.ModelCheckpoint("makerere.h5",save_weights_only=False,save_best_only=True)
    model.fit(train_ds,epochs=10,validation_data=val_ds,callbacks=[checkpoint])
    best = tf.keras.models.load_model("makerere.h5")
    val_loss,val_acc = best.evaluate(val_ds)
    print("\nVal Accuracy: {:.2f} %".format(100*val_acc))
    print("Val Loss: {:.2f} %".format(100*val_loss))