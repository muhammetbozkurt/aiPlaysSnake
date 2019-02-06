import numpy as np
from PIL import ImageGrab
import cv2
import time
from getkeys import key_check

#http://www.parkinggames.com/ok-parking.html
#https://tr.y8.com/games/parking
def screen_record(): 
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab(bbox=(200,230,850,400)))#200,230,850,600 for snake
        printscreen1 = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
       # printscreen1[(printscreen1[:,:]<150)]=255
       # printscreen1[(printscreen1[:,:])!=255]=0
        cv2.imshow('window',printscreen1)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



keysMatris = np.identity(8)

up = keysMatris[0]
left = keysMatris[1]
right = keysMatris[2]
down = keysMatris[3]
up_left = keysMatris[4]
up_right = keysMatris[5]
down_left = keysMatris[6]
down_right = keysMatris[7]

def keyTranslator(keys):
    print(keys)
    output = np.zeros(8)
    if("W" in keys):
        output = up
    elif("A" in keys):
        output = left
    elif("D" in keys):
        output = right
    elif("S" in keys):
        output = down
    if("W" in keys and "A" in keys):
        output = up_left
    elif("W" in keys and "D" in keys):
        output = up_right
    elif("S" in keys and "A" in keys):
        output = down_left
    elif("S" in keys and "D" in keys):
        output = down_right
    return output

def collecting_data():
    imageData = np.array([])
    keyData = np.array([])

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    notQuit = True
    count = 0
    while notQuit:
        screen =  np.array(ImageGrab.grab(bbox=(200,230,850,400)))#değiştirilmeli
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(224,224))
        screen[(screen[:,:]<100)]=255
        screen[(screen[:,:])!=255]=0        
        
        keyArray = keyTranslator(key_check())
        print("keyArray",keyArray)
        imageData = np.append(imageData,screen)
        keyData = np.append(keyData,keyArray)
        count+=1
        cv2.imshow("window",screen)#cv2.COLOR_BGR2RGB
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if(count%140 == 0):
            print("--------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------")
            file_name = "snakeData-%d" % (count+4)
            print(file_name," has been saved",)
            print(keyData.shape,imageData.shape)
            print("--------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------")
            np.save(file_name+"key.npy",keyData)
            np.save(file_name+"image.npy",imageData)
    keyData=keyData.reshape(-1,8)
    imageData=imageData.reshape(-1,224,224,1)
            
    print(keyData.shape,imageData.shape)

if __name__ == "__main__":
    
    screen_record()