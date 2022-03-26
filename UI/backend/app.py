from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['TESTING'] = False

CORS(app, support_credentials=True)

@app.route('/evaluate', methods=['POST'])
@cross_origin(supports_credentials=True)
def getEvaluation():

    input_json = request.get_json(force=True)

    result = np.random.choice([True, False])

    return result