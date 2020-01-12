import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


(x_train, y_train), (x_test, y_test) = mnist.load_data()   #loads MNIST data from keras 

num_train = x_train.shape[0]
num_test = x_test.shape[0]      #gives the number of training and test set examples
print (num_train, num_test)

index = 330
print (y_train[index])
plt.imshow(x_train[index])      #outputs an image and its value for checking if data is correctly imported

#NORMALIZATION PROCESS
x_train = x_train.reshape(num_train, 28, 28, 1)
x_test = x_test.reshape(num_test, 28, 28, 1)
img_shape = (28,28,1)
x_train = x_train*1.0/255
x_test = x_test*1.0/255    #multiplication by 1.0 ensures float value in the final result 

print (x_train.shape)
print (x_train.shape[0], x_test.shape[0])

model = Sequential()    #sequential object from keras

model.add(Conv2D(32, kernel_size=(3,3), input_shape=img_shape))     #convolution layer with filter of 3-by-3
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) 
model.add(Dense(128, activation=tf.nn.relu))        #applying non linear relu function
model.add(Dropout(rate=0.1))
model.add(Dense(10,activation=tf.nn.softmax))


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])         #using adam optimizer. rmsprop also gives good results
model.fit(x=x_train,y=y_train, epochs=10 )

model.evaluate(x_test, y_test)

index = 2133
plt.imshow(x_test[index].reshape(28, 28))
pred = model.predict(x_test[index].reshape(1, 28, 28, 1))
print(pred.argmax())        #predicting the value image corresponds to

correct_values = 0
for i in range(num_test):
    index=i    
    pred = model.predict(x_test[index].reshape(1, 28, 28, 1))
    if (pred.argmax()==y_test[index]):
        correct_values= correct_values+1

print ("Correct Test Values: ",correct_values,"/",num_test)         #gives number of correctly classified test values
