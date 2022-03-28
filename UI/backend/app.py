from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import numpy as np

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['TESTING'] = False

CORS(app, support_credentials=True)


@app.route('/evaluate', methods=['POST'])
@cross_origin(supports_credentials=True)
def getEvaluation():

    # input_json = request.get_json(force=True)

    result = np.random.choice([True, False])
    print(result)

    return jsonify({"evaluation": bool(result)}), 200


if __name__ == '__main__':
    app.run(port=9000)