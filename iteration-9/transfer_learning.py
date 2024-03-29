from os import PathLike
from numpy.core.fromnumeric import size
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from load_images_cnn import *
from utils import *
from keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="VGG16", help="model to be reused")
ap.add_argument("-s", "--size", default=64, help="image size")
ap.add_argument("-b", "--batch", default=32, help="batch size")
ap.add_argument("-e", "--epoch", default=30, help="number of epochs")
args = vars(ap.parse_args())

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, 
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

size = int(args['size'])
batch = int(args['batch'])
epochs = int(args['epoch'])

x, y = load_data(size) #loads images

#divide between train and test data
X_train, X_test, y_train, y_test =  train_test_split(x, y, test_size=0.33, random_state=84)

print('Loading model {0}...'.format(args['model']))

if args['model'].lower() == 'vgg16':
    model = VGG16(input_shape=(size,size,3), include_top = False, weights= 'imagenet')
elif args['model'].lower() == 'vgg19':
    model = VGG19(input_shape=(size,size,3), include_top = False, weights= 'imagenet')
elif args['model'].lower() == 'xception':
    model = Xception(input_shape=(size,size,3), include_top = False, weights= 'imagenet')

x = model.output
x = Flatten()(x)
x = Dense(3078,activation='relu')(x) 
x = Dropout(0.5)(x)
x = Dense(256,activation='relu')(x) 
x = Dropout(0.2)(x)
out = Dense(6,activation='softmax')(x)
tf_model=Model(inputs=model.input,outputs=out)

for layer in tf_model.layers[:20]:
    layer.trainable=False

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= r'C:\Users\fernando.fincatti\Documents\measuringMacroNutrients\iteration-9',
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    )

model_earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc",
    patience=10,
    mode="max",
    restore_best_weights=True,
)

tf_model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])
history = tf_model.fit(
    x=aug.flow(X_train, y_train, batch_size=batch),
    epochs = epochs,
    initial_epoch = 0,
    validation_data = (X_test, y_test),
    steps_per_epoch=len(X_train) // batch,
    callbacks=[model_checkpoint_callback, model_earlyStopping]
)

predict = predict_image(tf_model, X_test)
test = transform_y_test(y_test)
matrix = calculate_confusion_matrix(test, predict)

# evaluate the network
print("[INFO] evaluating network...")
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()