# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *

# %%
x_train, y_train, x_test, y_test, classes = load_dataset("./happy_cnn/datasets/train_happy.h5",
                                                         "./happy_cnn/datasets/test_happy.h5")
x_train, x_test = x_train / 255., x_test / 255.
y_train, y_test = y_train.T.squeeze(), y_test.T.squeeze()


# %%


def happy_model(input_shape):
    i = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(i)
    x = tf.keras.layers.Conv2D(32, (7, 7), 1, name='conv0')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name='bn0')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), name='max_pool')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='fc')(x)
    return tf.keras.models.Model(i, x)


# %%

model = happy_model((64, 64, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
# %%
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=40, batch_size=16)
# %%
print(model.summary())
# %%
plt.plot(r.history['loss'], label='train_loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
# %%
plt.plot(r.history['accuracy'], label='train_accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
# %%
img_path = 'happy_cnn/3.jpg'
img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
plt.imshow(img)

x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
plt.title('Happy' if model.predict(x)[0, 0] == 1 else "Sad")
plt.show()
print(model.predict(x))
# %%
