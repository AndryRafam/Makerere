import pandas as pd
from zipfile import ZipFile

train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")

def addextension(nm):
    return nm+".jpg"

train_df.Image_ID = train_df.Image_ID.apply(addextension)
test_df.Image_ID = test_df.Image_ID.apply(addextension)

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



from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import densenet, DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy

rescale = densenet.preprocess_input
base_model = DenseNet201(input_shape=(256,256,3),include_top=False,weights="imagenet")
base_model.trainable = False

class Transfer_DenseNet201():
    def model(self,input):
        self.x = rescale(input)
        self.x = base_model(self.x,training=False)
        self.x = GlobalAveragePooling2D()(self.x)
        self.x = Dropout(0.2,seed=42)(self.x)
        self.output = Dense(3,activation="softmax")(self.x)
        self.model = Model(input,self.output,name="Transfer_DenseNet201")
        return self.model

TFDES201 = Transfer_DenseNet201()
model = TFDES201.model(Input(shape=(256,256,3)))
model.compile(RMSprop(),CategoricalCrossentropy(),metrics=["accuracy"])
model.summary()

if __name__=="__main__":
    checkpoint = [
        ModelCheckpoint("makerere.hdf5",save_weights_only=False,save_best_only=True,monitor="val_accuracy")
    ]
    api.model.fit(train_ds,epochs=7,validation_data=val_ds,callbacks=checkpoint)
    best = load_model("makerere.hdf5")
    val_loss,val_acc = best.evaluate(val_ds)
    print("\nVal Accuracy: {:.2f} %".format(100*val_acc))
    print("Val Loss: {:.2f} %".format(100*val_loss))
