from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import sys

import logging

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
    im = Image.open(BytesIO(base64.b64decode(image_input)))
    print(im)

    result = np.random.choice([True, False])

    return jsonify({"evaluation": bool(result)}), 200


if __name__ == '__main__':
    app.run(port=9000)