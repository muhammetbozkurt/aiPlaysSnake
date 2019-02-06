from tensorflow.keras import models
import numpy as np 
import pickle
import cv2

model = models.load_model("snakeAI.model")

X = pickle.load(open("features.pickle","rb"))
y = pickle.load(open("label.pickle","rb"))

X = X.reshape(-1,224,224,1)
y = y.reshape(-1,8)
c = X[5]
c = c.reshape(224,224)
cv2.imshow("deneme",c)
cv2.waitKey()
cv2.destroyAllWindows()
X = X / 255

model.fit(X,y,batch_size=16,epochs=20,validation_split=0.1)

model.save("snakeAI.model")