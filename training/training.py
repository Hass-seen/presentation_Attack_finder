import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
import random



training_data=[]
attack_path = "C:/Users/Admin/Desktop/TF_pres_attack/training/attack/"
file_list = os.listdir(attack_path)
cout=0
for name in file_list:
    
    if "controlled" in name:
        continue
    print(name)
    with open(attack_path+name, "r") as f:
        data = json.load(f)
    if len(data)< 200:
        continue
    cout+=1
    if cout ==128:
        break
    training_data.append([data[0:200],1])
print(len(training_data))
real_path="C:/Users/Admin/Desktop/TF_pres_attack/training/real/"
real_list = os.listdir(real_path)
for name in real_list:
    with open(real_path+name, "r") as f:
        data = json.load(f)
    if len(data)< 240:
        continue
    training_data.append([data[0:200],0])
print(len(training_data))
random.shuffle(training_data)

print(len(training_data))


val_data= training_data[245:255]
training=training_data[0:244]
X=[]
Y=[]

for features, label in training:
    
    X.append(features)
    Y.append(label)
print(Y) 
print(len(X))  
Val_X=[]
Val_Y=[]
for features, label in val_data:
    Val_X.append(features)
    Val_Y.append(label)
 
 
X= np.array(X)   
Y= np.array(Y).astype('float32').reshape((-1,1))
Val_X= np.array(Val_X)   
Val_Y= np.array(Val_Y).astype('float32').reshape((-1,1))

model = keras.models.Sequential([
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(500, activation='relu'),
  keras.layers.Dense(1000, activation='relu'),
  keras.layers.Dense(500, activation='relu'),
  keras.layers.Dense(200, activation='relu'),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(64, activation='relu'),
#   keras.layers.Dropout(0.2),
  keras.layers.Dense(2, activation='softmax')
])

loss = tf.keras.losses.SparseCategoricalCrossentropy()
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 32
epoch = 30
model.fit(X, Y, batch_size=batch_size, epochs=epoch, shuffle=True, verbose=1)

preds = model.predict(Val_X)
predicted_class = np.argmax(preds, axis=1)

print("))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))")
outp=list(predicted_class)
print(outp)

# print("))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))")
test=[int(v[0]) for v in Val_Y.tolist()]
print(str(test))
count=0
for i in range(0, len(test)):
    if test[i] != outp[i]:
        count+=1
print(count)
# model.save('my_model.h5')
