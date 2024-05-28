import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.transform import resize
from sklearn.metrics import classification_report
import pickle

# Cargar los nombres de las clases
with open('eventos.pkl', 'rb') as f:
    eventos = pickle.load(f)

# Load model
evento_model = load_model("catastrofe.h5")

# Load new images
images = []
filenames = [
    'test/incendio.jpg',
    'test/inundacion.jpg',
    'test/asalto.jpg',
    'test/tornado.jpg'
]

for filepath in filenames:
    image = plt.imread(filepath)
    image_resized = resize(image, (28, 28), anti_aliasing=True, clip=False, preserve_range=True)
    images.append(image_resized)

X = np.array(images, dtype=np.uint8)
test_X = X.astype('float32')
test_X /= 255.

# Predict
predicted_classes = evento_model.predict(test_X)

# Print results
for i, img_tagged in enumerate(predicted_classes):
    predicted_label = np.argmax(img_tagged)
    print(f"{filenames[i]} -> {eventos[predicted_label]}")

