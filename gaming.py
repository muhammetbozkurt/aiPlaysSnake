from tensorflow.keras import models
import numpy as np 
from pynput.keyboard import Key,Controller
from PIL import ImageGrab
import cv2

model = models.load_model("snakeAI.model")

keyboard = Controller()

keysMatris = np.identity(8)

up = keysMatris[0]
left = keysMatris[1]
right = keysMatris[2]
down = keysMatris[3]
up_left = keysMatris[4]
up_right = keysMatris[5]
down_left = keysMatris[6]
down_right = keysMatris[7]

def keypress(key):
	if((key == np.argmax(up))):
		keyboard.press("w")
		print("w")
		keyboard.release("w")
	elif((key == np.argmax(left))):
		keyboard.press("a")
		print("a")
		keyboard.release("a")
	elif((key == np.argmax(right))):
		keyboard.press("d")
		print("d")
		keyboard.release("d")
	elif((key ==  np.argmax(down))):
		keyboard.press("s")
		print("s")
		keyboard.release("s")
	elif((key == np.argmax(up_left))):
		keyboard.press("w")
		keyboard.press("a")
		print("wa")
		keyboard.release("w")
		keyboard.release("a")
	elif((key == np.argmax(up_right))):
		keyboard.press("w")
		keyboard.press("d")
		print("wd")
		keyboard.release("w")
		keyboard.release("d")
	elif((key == np.argmax(down_left))):
		keyboard.press("s")
		keyboard.press("a")
		print("sa")
		keyboard.release("s")
		keyboard.release("a")
	elif((key == np.argmax(down_right))):
		keyboard.press("s")
		keyboard.press("d")
		print("sd")
		keyboard.release("s")
		keyboard.release("d")	
	else:
		print(" ")

notQuit = True
while notQuit:
    screen =  np.array(ImageGrab.grab(bbox=(200,230,850,600)))#değiştirilmeli
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen,(224,224))
    screen[(screen[:,:]<100)]=255
    screen[(screen[:,:])!=255]=0 
    screen = screen.reshape(-1,224,224,1)
    screen=screen/255
    key = model.predict(screen)
    key = np.argmax(key)
    keypress(key)

