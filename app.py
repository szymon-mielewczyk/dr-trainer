import io
import json
import torch
import torchvision
import numpy
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
from skimage import io as ski_io
app = Flask(__name__)

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


print(torchvision.__version__)
print(torch.__version__)

# model_path = 'resnet50_mias_26092023'
num_classes = 7
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# model.load_state_dict(torch.load(model_path))
model.eval()


def transform_image(image):
    image = cv2.merge((image,image,image))
    tensor = torch.tensor(image, dtype=torch.float32)
    tensor = torch.reshape(tensor, (1,3,1024,1024))
    tensor = tensor / 255
    return tensor


def get_prediction(image):
    tensor = transform_image(image)
    outputs = model(tensor)
    x = outputs[0]['boxes'][0][0] + (outputs[0]['boxes'][0][2] - outputs[0]['boxes'][0][0]) / 2
    y = outputs[0]['boxes'][0][1] + (outputs[0]['boxes'][0][3] - outputs[0]['boxes'][0][1]) / 2
    r = ((outputs[0]['boxes'][0][2] - outputs[0]['boxes'][0][0]) / 2) 
    x_point = str(int(x))
    y_point = str(int(y))
    radius = str(int(r))
    labels = str(annotations[int(outputs[0]['labels'][0])])
    score = str(float(outputs[0]['scores'][0]))
    return x_point, y_point, radius, labels, score


@app.route('/api/ai/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_name = str(request.files['imageName'].read(), encoding='utf-8')
        image_id = str(request.files['imageId'].read(), encoding='utf-8')
        image = ski_io.imread(img_name)
        x_point, y_point, radius, label, score = get_prediction(image)
        # todo add score threshold
        return jsonify({
            "imageId": image_id,
            "imageName": img_name,
            "medicalParams": {
                "abnormalityClass": label
            },
            "markedRegions": [
                {
                    "x": x_point,
                    "y": y_point,
                    "radius": radius
                }
            ]
        })


if __name__ == '__main__':
    app.run()
