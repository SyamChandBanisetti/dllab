import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load pretrained CNN model
model = MobileNetV2(weights='imagenet')

# Provide image path
img_path = "dog.png"


# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict object in image
preds = model.predict(img_array)
decoded = decode_predictions(preds, top=1)[0][0][1]

# Generate caption
caption = "A " + decoded + " is present in the image"

# Display image with caption
plt.imshow(img)
plt.title("Caption: " + caption)
plt.axis("off")
plt.show()

# Print caption
print("Generated Caption:", caption)