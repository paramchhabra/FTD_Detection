import tensorflow as tf
# from PIL import Image

def load_local_image(image, dim):
    image.seek(0)  # Important: Reset pointer to beginning
    img_bytes = image.read()
    
    if not img_bytes:
        raise ValueError("Image file is empty or could not be read.")
    
    img = tf.image.decode_image(img_bytes, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [dim, dim])
    img = tf.expand_dims(img, axis=0)
    return img


def predictftd(file):
    model = tf.keras.models.load_model("../Models/MainFtd.h5")
    prediction = model.predict(load_local_image(file,128))
    return prediction[0]

def predictmonth(file):
    model = tf.keras.models.load_model("../Models/FTD_Month_v1.h5")
    prediction = model.predict(load_local_image(file,64))
    return prediction[0]

# path = "C:\\Users\\param\\Downloads\\FTD_DATA\\Patients\\Month6\\124428.jpg"

# print(predictftd(path))
# print(predictmonth(path))