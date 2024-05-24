# import YOLO model
from ultralytics import YOLO

# Load a model
model = YOLO("runs/classify/train3/weights/best.pt") # load a pretrained model (recommended for training)

class_dict = {0: 'Angry', 1: 'Bored', 2: 'Confused', 3: 'cool', 4: 'Errrr', 5: 'Funny', 6: 'Happy', 7: 'Normal', 8: 'Proud', 9: 'Sad', 10: 'Scared', 11: 'Shy', 12: 'Sigh', 13: 'super_angry', 14: 'Surprised', 15: 'Suspicious', 16: 'sweet', 17: 'tricky', 18: 'unhappy', 19: 'Worried'}

class_predict = []
for i in test.image_ID:
    path = 'Face/train/Angry' + '/'+ i
    results = model.predict(path)
    class_predict.append(class_dict[results[0].probs.data.argmax().item()])


