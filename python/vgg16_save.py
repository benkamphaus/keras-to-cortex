from keras.applications.vgg16 import VGG16

m = VGG16()
with open("vgg16.json", "w") as f:
    f.write(m.to_json())
m.save_weights("vgg16.h5")
