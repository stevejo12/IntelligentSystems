import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

model = "sign.model"
model = load_model(model)
    
img = cv2.imread('LSign.jpg',0)
img = cv2.resize(img,(28, 28))

img = img.astype("float") / 255.0
img = image.img_to_array(img)
img = np.expand_dims(img, 0)

img = np.array([np.reshape(i, (28, 28)) for i in img])
plt.imshow(img[0], interpolation=None)
plt.show()

img = np.array([i.flatten() for i in img])
img = img.reshape(img.shape[0],28,28,1)

predict = model.predict(img)
result = np.argmax(predict)

maps= {0: "A", 1: "B", 2: "C", 3: "D",
       4: "E", 5: "F", 6: "G", 7: "H",
       8: "I", 9: "K", 10: "L", 11: "M",
       12: "N", 13: "O", 14: "P", 15: "Q",
       16: "R", 17: "S", 18: "T", 19: "U",
       20: "V", 21: "W", 22: "X", 23: "Y"
      }

if(result > 8):
    print("The sign refers to letter:",chr(result+1+65))
else:
    print("The sign refers to letter:",chr(result+65))
