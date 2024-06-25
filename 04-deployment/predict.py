import pickle
from flask import Flask, request, jsonify

with open('linear_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def prepare_features(ride_dto):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride_dto['PULocationID'], ride_dto['DOLocationID'])
    features['trip_distance'] = ride_dto['trip_distance']

    return features

def predict(features):
    X = dv.transform(features)
    predictions = model.predict(X)

    return predictions

app = Flask('trip-duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_controller():
    ride = request.get_json()
    print(ride)

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': float(pred[0])
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)