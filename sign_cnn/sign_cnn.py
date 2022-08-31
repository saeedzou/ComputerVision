#%%
import numpy as np
import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt
#%%
x_train, y_train, x_test, y_test, classes = load_dataset()
x_train, x_test = x_train/255., x_test/255.
y_train = tf.one_hot(y_train.T.squeeze(), depth=6).numpy()
y_test = tf.one_hot(y_test.T.squeeze(), depth=6).numpy()

#%%
# LeNet architecture
model = LeNet5((64, 64, 3), 6)
#%%
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
#%%

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)
#%%
plt.plot(r.history['loss'], label='train_loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
#%%
plt.plot(r.history['accuracy'], label='train_accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
