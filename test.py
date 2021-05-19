from tensorflow.keras.applications import ResNet50, DenseNet121

resnet = ResNet50(weights='imagenet')
resnet.summary()
 
 
json_string = resnet.to_json()
 
with open("./model.json", "w") as f : 
    f.write(json_string)