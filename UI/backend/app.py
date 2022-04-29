from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import sys

import logging

import torch
from torchvision import transforms, utils
import numpy as np

from models import *

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['TESTING'] = False

CORS(app, support_credentials=True)


@app.route('/evaluate', methods=['POST'])
@cross_origin(supports_credentials=True)
def getEvaluation():

    input_json = request.get_json(force=True)
    print("input_json", input_json['dataUrl'])
    image_input = input_json['dataUrl'].split(',')[1]
    # image_input = request.values['imageBase64']

    # img = Image.open(StringIO(image_input))

    # print("image_input: ", image_input)
    # print("base64: ", base64.b64decode(image_input))

    image = Image.open(BytesIO(base64.b64decode(image_input)))

    # img = image.resize((28, 28), Image.ANTIALIAS)
    # # pixels.shape == (28, 28, 4)
    # pixels = np.asarray(img, dtype='uint8')
    # # force (28, 28)
    # pixels = np.resize(pixels, (28, 28))
    # # image is distorted
    # img = Image.fromarray(pixels)
    # img.show()

    # image.show()

    # image.save(BytesIO(), 'PNG')

    # im = torch.from_numpy(
    #     np.array(Image.open(BytesIO(base64.b64decode(image_input))))).float()

    im = Image.open(BytesIO(base64.b64decode(image_input))).convert("RGBA")
    im = im.resize((28, 28))
    new_im = Image.new("RGBA", im.size, "WHITE")
    new_im.paste(im, mask=im)
    im_rgb = new_im.convert('RGB').convert('L')
    im_rgb.show()

    im = np.array(im_rgb)
    size = min(im.shape[0], im.shape[1])
    im = im[:size, :size]
    im = torch.from_numpy(im).float()
    # print(im)
    print('initial img size', im.shape)

    transform = transforms.Compose([
        # transforms.Resize((28, 28)),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
        transforms.Normalize((0.5, ), (0.5, )),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    input_image = transform(im)
    input_image = input_image[None, :]

    print("input_image: ", input_image)
    print('size', input_image.shape)
    model = MLP(28 * 28, 32, 11, 2)
    model.load_state_dict(
        torch.load(
            '../../classifiers/model_weights/mlp/mlp18_Apr_2022_19_17_00.pth'))

    model.eval()
    preds = model(input_image)
    print(f'preds probabilities: {preds}')
    _, prediction_result = torch.max(preds, 1)

    print("prediction_result: ", prediction_result)

    # result = np.random.choice([True, False])

    return jsonify({"evaluation": prediction_result.item()}), 200


if __name__ == '__main__':
    app.run(port=9000)