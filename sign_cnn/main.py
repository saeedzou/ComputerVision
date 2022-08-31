import numpy as np
import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test, classes = load_dataset("./datasets/train_signs.h5",
                                                         "./datasets/test_signs.h5")
x_train, x_test = x_train/255., x_test/255.
y_train = tf.one_hot(y_train.T.squeeze(), depth=6).numpy()
y_test = tf.one_hot(y_test.T.squeeze(), depth=6).numpy()

# Train with LeNet architecture
model = LeNet5((64, 64, 3), 6)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

# Plot training and validation loss and accuracy plotss
plot_loss_accuracy_vs_epoch(r, './sign_cnn/loss_acc_plot.png')
# Test with arbitrary image
img_path = './2.jpg'  # change this to your image path
img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
plt.imshow(img)

d = tf.keras.utils.img_to_array(img)
d = np.expand_dims(d, axis=0)
d = tf.keras.applications.resnet50.preprocess_input(d)
plt.title(f'Predicted: {np.argmax(model.predict(d)[0])}')
plt.show()
print(model.predict(d)[0])