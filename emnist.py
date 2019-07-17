from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

datasets = list_datasets()		#imports emnist datasets
x_train, y_train = extract_training_samples('byclass')		#saves the "byclass" training examples to x_train and y_train
x_test, y_test = extract_test_samples('byclass')
print (x_train.shape)	

num_train = x_train.shape[0]

num_test = x_test.shape[0]

print (num_train, num_test)
print (x_train.shape)

index = 11000
print (y_train[index])
plt.imshow(x_train[index])      #outputs an image and its value for checking if data is correctly imported

x_train = x_train.reshape(num_train, 28, 28, 1)
x_test = x_test.reshape(num_test, 28, 28, 1)
img_shape = (28,28,1)
x_train = x_train*1.0/255
x_test = x_test*1.0/255    #multiplication by 1.0 ensures float value in the final result 

print (x_train.shape)
print (x_train.shape[0], x_test.shape[0])





model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1', input_shape=img_shape))
model.add(MaxPooling2D((2, 2), name='maxpool_1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
model.add(MaxPooling2D((2, 2), name='maxpool_2'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
model.add(MaxPooling2D((2, 2), name='maxpool_3'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', name='dense_1'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu', name='dense_2'))
model.add(Dense(62, activation='sigmoid', name='output'))		

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])         #using adam optimizer
model.fit(x=x_train,y=y_train, epochs=10 )

model.evaluate(x_test, y_test)

index = 2453
plt.imshow(x_test[index].reshape(28, 28))
pred = model.predict(x_test[index].reshape(1, 28, 28, 1))
print(pred.argmax())        #predicting the value image corresponds to
