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

prompt_dict = {
    0: 'The Eiffel Tower',
    1: 'The Great Wall of China',
    2: 'airplane',
    3: 'angel',
    4: 'animal migration',
    5: 'basket',
    6: 'bread',
    7: 'candle',
    8: 'cloud',
    9: 'diving board',
    10: 'dog',
    11: 'zigzag',
    12: 'tree',
    13: 'wine glass',
    14: 'firetruck',
    15: 'eye',
    16: 'flower',
    17: 'guitar',
    18: 'toilet',
    19: 'ice cream',
    20: 'table',
    21: 'duck',
    22: 'knee',
    23: 'elephant',
    24: 'tornado',
    25: 'hexagon',
    26: 'umbrella',
    27: 'triangle',
    28: 'hand',
    29: 'violin'
}

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
    im = im.resize((224, 224))
    new_im = Image.new("RGBA", im.size, "WHITE")
    new_im.paste(im, mask=im)
    im_rgb = new_im.convert('RGB').convert('L')
    # im_rgb.show()

    im = np.array(im_rgb)
    im = np.invert(im)
    size = min(im.shape[0], im.shape[1])
    im = im[38:262, :224]
    im = torch.from_numpy(im).float()

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.expand((3, -1, -1))),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # transform = transforms.Compose([
    #     transforms.Lambda(lambda x: x.expand((3, -1, -1))),
    #     transforms.Resize((28, 28)),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    input_image = transform(im)
    input_image = input_image[None, :]

    # print("input_image: ", input_image)
    print('size', input_image.shape)

    # model = CNN(3, 32, 30, 3, 0.25, 0.5)
    # model.load_state_dict(
    #     torch.load(
    #         '../../classifiers/model_weights/cnn/cnn05_May_2022_08_54_11.pth'))
    # model.load_state_dict(
    #     torch.load(
    #         '../../classifiers/model_weights/cnn/cnn29_Apr_2022_21_54_43.pth'))

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

    result = prompt_dict[prediction_result.item()]

    print("prediction_result: ", result)

    # result = np.random.choice([True, False])

    return jsonify({"evaluation": result}), 200


if __name__ == '__main__':
    app.run(port=9000)
