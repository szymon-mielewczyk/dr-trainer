import io
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2

app = Flask(__name__)

print(torchvision.__version__)
print(torch.__version__)

# model_path = 'resnet50_mias_29092023'
num_classes = 7
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# model.load_state_dict(torch.load(model_path))
model.eval()

annotations = {
            'CALC': 1,
            'CIRC': 2,
            'ARCH': 3,
            'SPIC': 4,
            'MISC': 5,
            'ASYM': 6,
             1: 'CALC',
             2: 'CIRC',
             3: 'ARCH',
             4: 'SPIC',
             5: 'MISC',
             6: 'ASYM',
        }

def transform_image(image):
    image = Image.open(io.BytesIO(image))
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(1024,antialias=True)])
    tensor = my_transforms(image).unsqueeze(0)
    image = tensor.numpy()
    if image.shape == (1,1,1024,1024):
        image = cv2.merge((image,image,image))
    tensor = torch.tensor(image, dtype=torch.float32)
    tensor = torch.reshape(tensor, (1,3,1024,1024))
    tensor = tensor / 255
    return tensor


def get_prediction(image):
    tensor = transform_image(image)
    outputs = model(tensor)
    boxes = str(outputs[0]['boxes'][0].tolist())
    labels = str(annotations[int(outputs[0]['labels'][0])])
    score = str(float(outputs[0]['scores'][0]))
    return boxes, labels, score


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['file'].read()
        bbox, labels, score = get_prediction(img)
        return jsonify({'bbox': bbox, 'label': labels, 'score':score})


if __name__ == '__main__':
    app.run()