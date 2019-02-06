import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Activation,ZeroPadding2D,Dropout
from tensorflow.keras import optimizers
import pickle
import numpy as np
import matplotlib.pyplot as plt
"""
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,1)))#değiştirdim
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(8, activation='softmax'))    

    if weights_path:
        model.load_weights(weights_path)

    return model
"""
def alternative_model():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224,1)))#225
	model.add(Conv2D(64,(3,3),activation="relu"))#223
	model.add(ZeroPadding2D((1,1)))#224
	model.add(Conv2D(128,(3,3),activation="relu"))#222
	model.add(MaxPooling2D((3,3)))#74

	model.add(ZeroPadding2D((1,1)))#75
	model.add(Conv2D(128,(3,3),activation="relu"))#72
	model.add(ZeroPadding2D((1,1)))#73
	model.add(Conv2D(128,(4,4),activation="relu"))#70
	model.add(MaxPooling2D(7,7))

	model.add(Flatten())
	model.add(Dense(1200,activation="relu"))
	model.add(Dense(512,activation="relu"))
	model.add(Dense(64,activation="relu"))
	model.add(Dense(8,activation="softmax"))

	return model


x = pickle.load(open("features.pickle","rb"))
y = pickle.load(open("label.pickle","rb"))
x = x.reshape(-1,224,224,1)
y = y.reshape(-1,8)
print(y.shape)
print(x.shape[1:])
print(y[0])
c = x[5]
c = c.reshape(224,224)
plt.imshow(c)
plt.show()
x/=255

model = alternative_model()
sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd,
	loss="mse",
	metrics=["accuracy"])

model.fit(x,y,batch_size=2,epochs=15,validation_split=0.1)

model.save("snakeAI.model")

