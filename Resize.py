import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    img = Image.open(image_path)

    img = img.convert('L')

    img = img.resize((28, 28))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = img_array.flatten()

    img_array = img_array.reshape(784, 1)

    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    plt.show()

    return img_array

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import Network
net = Network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

image_path = 'number.png'
processed_image = preprocess_image(image_path)

prediction = np.argmax(net.feedforward(processed_image))

print(f"Predicted digit: {prediction}")
