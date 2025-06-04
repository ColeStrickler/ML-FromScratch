import struct
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt

import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}")
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape((num_images, rows, cols))
    return images

def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in label file {filename}")
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = list(f.read(num_labels))
    return labels




images = load_mnist_images("t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
n = 1233  # change this to any index < 10000


labels = read_mnist_labels('t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
print(f"First label: {labels[1233]}")  # Prints the label (digit) for first image
plt.imshow(images[n], cmap="gray")
plt.title(f"Image #{n}")
plt.axis("off")
plt.show()