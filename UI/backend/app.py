from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import sys

import logging

import torch
from torchvision import transforms, utils, models
import numpy as np

from models import *

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['TESTING'] = False

CORS(app, support_credentials=True)


@app.route('/evaluate', methods=['POST'])
@cross_origin(supports_credentials=True)
def getEvaluation():

    input_json = request.get_json(force=True)
    # print(input_json)
    image_input = input_json['dataUrl'].split(',')[1]
    # print(image_input)

    im = Image.open(BytesIO(base64.b64decode(image_input))).convert("RGBA")
    im = im.resize((28, 28))
    new_im = Image.new("RGBA", im.size, "WHITE")
    new_im.paste(im, mask=im)
    im_rgb = new_im.convert('RGB').convert('L')
    # im_rgb.show()

    im = np.array(im_rgb)
    size = min(im.shape[0], im.shape[1])
    im = im[:size, :size]
    im = torch.from_numpy(im).float()

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.expand((3, -1, -1))),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_image = transform(im)
    input_image = input_image[None, :]

    # print("input_image: ", input_image)
    print('size', input_image.shape)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 30)
    model.load_state_dict(
        torch.load(
            '../../classifiers/model_weights/resnet/resnet05_May_2022_11_46_38.pth',
            map_location=torch.device('cpu')))

    model.eval()
    preds = model(input_image)
    print(f'preds probabilities: {preds}')
    _, prediction_result = torch.max(preds, 1)

    print("prediction_result: ", prediction_result)

    # result = np.random.choice([True, False])

    return jsonify({"evaluation": prediction_result.item()}), 200


if __name__ == '__main__':
    app.run(port=9000)