import numpy as np
import pickle


bigDataİmage=[]
bigDataKey = []
  
for i in range(3):
	namei = "snakeData-%dimage.npy" %((i+1)*140)
	namek = "snakeData-%dkey.npy" %((i+1)*140)
	print(namei)
	print(namek)
	tempİmageArray = np.load(namei)
	tempKeyArray = np.load(namek)
	print("tempKeyArray",tempKeyArray)
	print("tempİmageArray",tempİmageArray)

	bigDataİmage = np.append(bigDataİmage,tempİmageArray.reshape(-1,224,224,1))
	bigDataKey = np.append(bigDataKey,tempKeyArray.reshape(-1,8))
	

print("bigDataKey",bigDataKey)


out_pickle = open("features.pickle","wb")
pickle.dump(bigDataİmage,out_pickle)
out_pickle.close()

out_pickle = open("label.pickle","wb")
pickle.dump(bigDataKey,out_pickle)
out_pickle.close()
