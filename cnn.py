import numpy as np
import pickle
import cv2
x = pickle.load(open("features.pickle","rb"))
y = pickle.load(open("label.pickle","rb"))
x = x.reshape(-1,224,224,3)
y = y.reshape(-1,8)

print(y.shape)
y = y[:420:]
print(y.shape)