import win32api as wapi
import numpy as np


keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys+=key
    return keys


if __name__ == "__main__":
	for _ in range(1000):
		print(keyTranslator(key_check()))

	print(["a","b","c","d"])