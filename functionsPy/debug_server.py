from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

@app.route('/save-tensor', methods=['POST'])
def save_tensor():
    print('test_data received')
    tensor_data = request.get_json()  # Get JSON data from request
    print(tensor_data)
    if tensor_data is None:
        return jsonify({'error': 'No data provided'}), 400

    # Write JSON data to a file
    with open('tensorData.json', 'w') as f:
        json.dump(tensor_data, f)

    return jsonify({'message': 'Data saved successfully'}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)