import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Load the model from your local hard drive
print("Loading model...")
model = tf.keras.models.load_model('best_violence_classifier_finetuned.keras')

# 2. Point this to the image you want to test
img_path = 'test.jpg' 

# 3. Process the image to match what the AI expects
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# 4. Make the prediction
print("Scanning image...")
prediction = model.predict(x)
score = prediction[0][0]

# 5. Output the result
if score > 0.5:
    print(f"\n⚠️ Prediction: VIOLENT")
    print(f"Confidence: {score * 100:.2f}%")
else:
    print(f"\n✅ Prediction: NON-VIOLENT")
    print(f"Confidence: {(1 - score) * 100:.2f}%")