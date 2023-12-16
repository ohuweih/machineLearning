import numpy
import matplotlib.pyplot
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50

#Load Already trained model
model = ResNet50(weights='imagenet')

#Load and preprocess Image
img_path = "./chips.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = numpy.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make Predictions
predictions = model.predict(img_array)

#print(predictions)

#decode predictions
decoded_predictions = decode_predictions(predictions, top=3)[0]
print(decoded_predictions)
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")

#matplotlib.pyplot.imshow(img)
#matplotlib.pyplot.show()
