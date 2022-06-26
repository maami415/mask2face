import matplotlib.pyplot as plt
from utils.configuration import Configuration
from utils.data_generator import DataGenerator
from utils.model import Mask2FaceModel

configuration = Configuration()
dg = DataGenerator(configuration)

model = Mask2FaceModel.load_model('models/model.h5')
input_img = 'data/test.jpg'
generated_output = model.predict(input_img)

plt.imshow(generated_output)
plt.axis("off")
plt.show()