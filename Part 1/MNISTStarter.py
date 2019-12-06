# Igor Pieters
# UCID: 30061116
# CPSC 501 - Fall 2019
# Assignment 1 - Part 1

import tensorflow as tf

print("--Get data--")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("--Make model--")
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("--Fit train model--")
model.fit(x_train, y_train, epochs=10, verbose=2)


print("--Evaluate train model--")
model_loss, model_acc = model.evaluate(x_train,  y_train, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

print("--Evaluate test model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")
#Save Model
model.save('MNIST.h5')