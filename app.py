from flask import Flask, request, jsonify
from model_utils import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    symbol = data.get('symbol')
    features = data.get('features')

    if not symbol or not features:
        return jsonify({'error': 'symbol and features are required'}), 400

    try:
        result = predict(symbol, features)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)

