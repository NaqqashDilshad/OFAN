import os
import datetime
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# import model and initialize the number of training epochs and batch size
from tensorflow.keras.applications import MobileNetV3Small as XModel
num_epochs = 10
batch_size = 32
input_size = 224
num_classes = 2
resultSavePath='ResultsMobileNetV3Small/'


# derive the path to the directories containing the training,
# validation, and testing splits, respectively
TRAIN_PATH = os.path.sep.join(["NewSplittedDataset", "training"])
VAL_PATH = os.path.sep.join(["NewSplittedDataset", "validation"])
TEST_PATH = os.path.sep.join(["NewSplittedDataset", "testing"])


# determine the total number of image paths in training, validation, and testing directories
totalTrain=len(TRAIN_PATH)
totalVal=len(VAL_PATH)
totalTest=len(TEST_PATH)
print('totalTrain ==>',totalTrain)
print('totalVal ==>',totalVal)
print('totalTest ==>',totalTest)


# initialize the training training data augmentation object
trainAug=ImageDataGenerator(
rotation_range=20,
zoom_range=0.01,
width_shift_range=0.01,
height_shift_range=0.01,
shear_range=0.01,
horizontal_flip=True,
vertical_flip=True,
fill_mode="nearest")


# initialize the validation  data augmentation object                                                
valAug=ImageDataGenerator()
# initialize the  testing data augmentation object
testAug=ImageDataGenerator() 


# initialize the training generator
trainGen=trainAug.flow_from_directory(           TRAIN_PATH,
                                                 target_size=(input_size, input_size),
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=True)

# initialize the validation generator
valGen = valAug.flow_from_directory(             VAL_PATH,
                                                 target_size=(input_size, input_size),
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=False)

# initialize the testing generator
testGen=testAug.flow_from_directory(             TEST_PATH,
                                                 target_size=(input_size, input_size),
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=False)


################ Model ################
inputs = tensorflow.keras.Input(shape=(input_size, input_size, 3))
base_model = XModel(weights='imagenet', input_shape=(input_size, input_size, 3), include_top=False)
base_model_input = base_model(inputs)
x3=layers.GlobalAveragePooling2D()(base_model_input)
x=layers.BatchNormalization()(base_model_input)


################ Channel Attention ################
x1=layers.GlobalAveragePooling2D()(x)
x1=layers.Dense(100, activation='relu')(x1)
x1=layers.Dense(50, activation='relu')(x1)
x1=layers.Dense(25, activation='relu')(x1)
x1=layers.BatchNormalization()(x1)
print("output of channel info ", x1)


################ Spatial Attention ################
x2=layers.Conv2D(filters = 64,kernel_size = (1,1), dilation_rate=1, activation='relu', padding='same')(x)
x2=layers.Conv2D(filters = 64,kernel_size = (3,3), dilation_rate=2, activation='relu', padding='same')(x2)
x2=layers.Conv2D(filters = 64,kernel_size = (1,1), dilation_rate=3, activation='relu', padding='same')(x2)
x2=layers.GlobalAveragePooling2D()(x2)
x2=layers.BatchNormalization()(x2)
print("output of Spatial info ", x2)


################ BAM ################
BAM=layers.concatenate([x1, x2])
BAM=layers.BatchNormalization()(BAM)
print("output of Final BAM ", BAM)
BAM=layers.concatenate([x3, BAM])
F=layers.Dense(150, activation='relu')(x3)
F=layers.BatchNormalization()(F)
outputs=layers.Dense(units=num_classes, activation='softmax')(F)
model = tensorflow.keras.Model(inputs, outputs)


opt=Adam(learning_rate=0.0001,)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
model.summary()
a = datetime.datetime.now()
history = model.fit(trainGen,validation_data=valGen, epochs= num_epochs)
print('Model saving')    
model.save(resultSavePath+'Model+CA+SA.h5')