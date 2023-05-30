#READING AND UNDERSTANDING QR CODE
import cv2, os
from pyzbar.pyzbar import decode
from matplotlib import pyplot as plt
from _tkinter import *
import pyttsx3
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

capture = cv2.videocapture(0)  #DEFAULT CAMERA

while True:
    _, instantpic = capture.read()

    decodeddata = decode(instantpic)
    NETDATA = decodeddata[0][0] #FOR ACCESING THE TEXT PART ONLY

    cv2.imshow("qr",instantpic)
    key = cv2.waitKey(5)

    if key == ord(POWER HANDLE): #FIRST RED SWITCH
        break

#to read out text



engine = pyttsx3.init()
engine.say(NETDATA)

engine.runAndWait()

#FOR CAPTURING EMERGENCY PIC AND SAVING

emergencypic = cv2.videocapture(0)

a = 0
while True :
    a = a +1
    check , frame = emergencypic.read()
    cv2.waitKey(0)
    frame.png("emergency pic")
    if key == ord("EMERGENCY BUTTON")  #THE SECOND RED BUTTON

#FACEMASK RECOGNITON


        data_path = 'dataset'
        categories = os.listdir(data_path)
        labels = [i for i in range(len(categories))]

        label_dict = dict(zip(categories, labels))  # empty dictionary

        print(label_dict)
        print(categories)
        print(labels)
{'with mask': 0, 'without mask': 1}
['with mask', 'without mask']
[0, 1]

img_size = 100
data = []
target = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #MAKING IT GREY AND 10*10 TO MAKE IT IT EASIER TO DIGEST FOR TRAINING
        data.append(resized)
        target.append(label_dict[category])



data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)

data=np.load('data.npy')
target=np.load('target.npy')

#NEURAL NETWORK

model=Sequential()

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(50,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

 #TRAINING WITH THE DATA
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)
Train on 990 samples, validate on 248 samples
Epoch 1/20
990/990 [==============================] - 93s 94ms/step - loss: 0.7326 - accuracy: 0.5626 - val_loss: 0.5822 - val_accuracy: 0.6290
Epoch 2/20
990/990 [==============================] - 93s 94ms/step - loss: 0.5465 - accuracy: 0.7253 - val_loss: 0.4429 - val_accuracy: 0.8185
Epoch 3/20
990/990 [==============================] - 93s 94ms/step - loss: 0.3708 - accuracy: 0.8354 - val_loss: 0.2568 - val_accuracy: 0.9032
Epoch 4/20
990/990 [==============================] - 95s 96ms/step - loss: 0.2679 - accuracy: 0.8970 - val_loss: 0.1807 - val_accuracy: 0.9476
Epoch 5/20
990/990 [==============================] - 93s 94ms/step - loss: 0.1917 - accuracy: 0.9303 - val_loss: 0.2207 - val_accuracy: 0.9315
Epoch 6/20
990/990 [==============================] - 93s 94ms/step - loss: 0.1749 - accuracy: 0.9343 - val_loss: 0.1249 - val_accuracy: 0.9597
Epoch 7/20
990/990 [==============================] - 95s 96ms/step - loss: 0.1238 - accuracy: 0.9576 - val_loss: 0.1258 - val_accuracy: 0.9637
Epoch 8/20
990/990 [==============================] - 94s 95ms/step - loss: 0.1037 - accuracy: 0.9616 - val_loss: 0.1243 - val_accuracy: 0.9516
Epoch 9/20
990/990 [==============================] - 94s 95ms/step - loss: 0.0893 - accuracy: 0.9687 - val_loss: 0.1095 - val_accuracy: 0.9556
Epoch 10/20
990/990 [==============================] - 94s 95ms/step - loss: 0.0540 - accuracy: 0.9828 - val_loss: 0.1193 - val_accuracy: 0.9597
Epoch 11/20
990/990 [==============================] - 92s 93ms/step - loss: 0.0399 - accuracy: 0.9899 - val_loss: 0.1278 - val_accuracy: 0.9677
Epoch 12/20
990/990 [==============================] - 93s 94ms/step - loss: 0.0518 - accuracy: 0.9818 - val_loss: 0.0974 - val_accuracy: 0.9718
Epoch 13/20
990/990 [==============================] - 96s 97ms/step - loss: 0.0615 - accuracy: 0.9778 - val_loss: 0.1604 - val_accuracy: 0.9274
Epoch 14/20
990/990 [==============================] - 97s 98ms/step - loss: 0.0589 - accuracy: 0.9828 - val_loss: 0.0863 - val_accuracy: 0.9597
Epoch 15/20
990/990 [==============================] - 94s 95ms/step - loss: 0.0411 - accuracy: 0.9808 - val_loss: 0.0998 - val_accuracy: 0.9677
Epoch 16/20
990/990 [==============================] - 81s 82ms/step - loss: 0.0547 - accuracy: 0.9747 - val_loss: 0.0899 - val_accuracy: 0.9556
Epoch 17/20
990/990 [==============================] - 79s 80ms/step - loss: 0.0372 - accuracy: 0.9889 - val_loss: 0.0855 - val_accuracy: 0.9637
Epoch 18/20
990/990 [==============================] - 78s 79ms/step - loss: 0.0301 - accuracy: 0.9879 - val_loss: 0.1107 - val_accuracy: 0.9556
Epoch 19/20
990/990 [==============================] - 89s 90ms/step - loss: 0.0206 - accuracy: 0.9919 - val_loss: 0.0947 - val_accuracy: 0.9677
Epoch 20/20
990/990 [==============================] - 94s 95ms/step - loss: 0.0358 - accuracy: 0.9899 - val_loss: 0.1575 - val_accuracy: 0.9476

#PLOTTING THE NEURAL THE DATA OF ACCURACY FROM TRAING LOSS DATA STRUCTURE
plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

print(model.evaluate(test_data,test_target))
138/138 [==============================] - 6s 44ms/step
[0.14019694376358952, 0.9637681245803833]

#RECIPROCATE THE TRAINED NETWORK AGAINST REALTIME CAMERA PLOT
model = load_model('model-017.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(2)

labels_dict={0:'MASK',1:'NO MASK'}


while (True):

    ret, img = source.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        if labels_dict = 1:
           import pyttsx3
         engine.say("PLEASE KEEP DISTANCE NO MASK")


     key = cv2.waitKey(2)
     if key == ord("POWER HANDLE"):
        BREAK


