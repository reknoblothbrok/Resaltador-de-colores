import cv2
import numpy as np
import imutils
import pandas as pd
#Establesco el rango de mínimos y máximos HSV
#Rojo
rojoBajo1 = np.array([0, 140, 90], np.uint8)
rojoAlto1 = np.array([8, 255, 255], np.uint8)
rojoBajo2 = np.array([160, 140, 90], np.uint8)
rojoAlto2 = np.array([180, 255, 255], np.uint8)

#Verde
verdeBajo = np.array([25,0,0], np.uint8)
verdeAlto = np.array([85, 255, 255], np.uint8)

# Leer la imagen
image = cv2.imread('IMG100.jpg')
image = imutils.resize(image, width=720)

# Pasamos las imágenes de BGR a: GRAY (esta a BGR nuevamente) y a HSV
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imageGray = cv2.cvtColor(imageGray, cv2.COLOR_GRAY2BGR)
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Detectamos el color rojo
redMask1 = cv2.inRange(imageHSV, rojoBajo1, rojoAlto1)
redMask2 = cv2.inRange(imageHSV, rojoBajo2, rojoAlto2)
rmask = cv2.add(redMask1, redMask2)
rmask = cv2.medianBlur(rmask, 7)
redDetected = cv2.bitwise_and(image,image,mask=rmask)

#Para el conteo de pixeles rojos
ret,BINARY = cv2.threshold(rmask,127,255,cv2.THRESH_BINARY)
redPixels = cv2.countNonZero(BINARY)

# Fondo en grises
invMask = cv2.bitwise_not(rmask)
bgGray = cv2.bitwise_and(imageGray,imageGray,mask=invMask)

# Sumamos bgGray y redDetected
finalImage = cv2.add(bgGray,redDetected)

#Detectamos el color verde
greenMask = cv2.inRange(imageHSV, verdeBajo, verdeAlto)
greenDetected = cv2.bitwise_and(image, image, mask=greenMask)

#Para el conteo de pixeles verdes
ret,BINARY2 = cv2.threshold(greenMask,127,255,cv2.THRESH_BINARY)
greenPixels = cv2.countNonZero(BINARY2)

#Fondo gris
invMask2 = cv2.bitwise_not(greenMask)
bgGray = cv2.bitwise_and(imageGray, imageGray, mask=invMask2)

#Sumamos bgGray y greenDetected
finalImage2 = cv2.add(bgGray, greenDetected)

#Export to CSV file
data = {'Text': ['Number of red pixels','Number of green pixels'],
        'Data':[redPixels, greenPixels]}
df = pd.DataFrame(data, columns=['Text', 'Data'])
df.to_csv('filename.csv')
# Visualización
print("Number of red pixels: ", redPixels)
print("Number of green pixels: ", greenPixels)
print(df)
cv2.imshow('Image',image)
cv2.imshow('Red', finalImage)
cv2.imshow('Green', finalImage2)
cv2.waitKey(0)
cv2.destroyAllWindows()