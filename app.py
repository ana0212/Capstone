import os
import joblib
import json
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, Model, IntegerField,
    FloatField, TextField, BooleanField
)
from playhouse.db_url import connect
from data_cleaning import clean_data

########################################
# Begin database stuff

app = Flask(__name__)

# Configuração do banco de dados
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = BooleanField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model

with open('columns.json') as fh:
    columns = json.load(fh)

with open('dtypes.pkl', 'rb') as fh:
    dtypes = pickle.load(fh)

pipeline = joblib.load('pipeline.pkl')

# End model un-pickling
########################################

########################################
# Begin webserver stuff

@app.route('/will_recidivate/', methods=['POST'])
def will_recidivate():
    try:
        request_data = request.get_json()
    except Exception as e:
        return jsonify({"error": "Failed to decode JSON object"}), 400

    # Function to validate a number
    def is_within_range(value, min_value, max_value):
        return min_value <= value <= max_value

    # Function to validate numerical fields
    def validate_numerical_field(value, field_name, min_value, max_value):
        if pd.isna(value):
            return None  # Permitir valores NaN
        if not isinstance(value, (int, float)):
            return f"Invalid value '{value}' for '{field_name}'"
        elif not is_within_range(value, min_value, max_value):
            return f"Value '{value}' for '{field_name}' out of range [{min_value}, {max_value}]"
        else:
            return None

    # Function to validate categorical fields
    def validate_categorical_field(value, field_name, valid_values):
        if pd.isna(value):
            return None  # Permitir valores NaN
        if value not in valid_values:
            return f"Invalid value '{value}' for '{field_name}'"
        return None

    # Function to validate datetime fields
    def validate_datetime(value, field_name):
        if pd.isna(value):
            return None  # Permitir valores NaN
        try:
            pd.to_datetime(value)
            return None
        except ValueError:
            return f"Invalid datetime format for '{field_name}'"

    response = {}

    observation_id = request_data.get("id", None)
    if observation_id is None:
        response["error"] = "Missing observation_id"
        return jsonify(response), 400

    # Verify if ID already exists
    if Prediction.select().where(Prediction.observation_id == observation_id).exists():
        response["error"] = "Observation ID already exists"
        return jsonify(response), 400

    # Validate unexpected columns
    expected_fields = set(["id", "name", "sex", "dob", "race", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count", "c_case_number", "c_charge_degree", "c_charge_desc", "c_offense_date", "c_arrest_date", "c_jail_in"])
    received_fields = set(request_data.keys())
    unexpected_fields = received_fields - expected_fields
    if unexpected_fields:
        response["error"] = f"Unexpected fields: {', '.join(unexpected_fields)}"
        return jsonify(response), 400

    # Validate fields
    data = request_data

    # Ensure all expected fields are present
    for field in expected_fields:
        if field not in data:
            data[field] = None

    # Validate categorical fields
    valid_categories = {
        'sex': ['Male', 'Female'],
        'c_charge_degree': ['F', 'M'],
        'race': ['Caucasian', 'African-American', 'Other', 'Hispanic', 'Native American', 'Asian']
    }
    
    for feature, valid_values in valid_categories.items():
        if feature in data:
            error_message = validate_categorical_field(data[feature], feature, valid_values)
            if error_message:
                response["error"] = error_message
                return jsonify(response), 400

    # Validate numerical fields
    numerical_features = {"juv_fel_count": (0, 50), "juv_misd_count": (0, 50), "juv_other_count": (0, 50), "priors_count": (0, 50)}  
    for feature, (min_value, max_value) in numerical_features.items():
        if feature in data:
            error_message = validate_numerical_field(data[feature], feature, min_value, max_value)
            if error_message:
                response["error"] = error_message
                return jsonify(response), 400

    # Validate datetime fields
    datetime_features = ["dob", "c_offense_date", "c_arrest_date", "c_jail_in"]
    for feature in datetime_features:
        if feature in data:
            error_message = validate_datetime(data[feature], feature)
            if error_message:
                response["error"] = error_message
                return jsonify(response), 400

    # Convert and clean the entries
    try:
        # Aplicar transformações do pipeline
        df = pd.DataFrame([data])
        df = clean_data(df)
        probability = pipeline.predict_proba(df)[0][1]
    
        # Aplicar o threshold de 0.6
        threshold = 0.6
        prediction = probability >= threshold

        # Map prediction to True or False
        prediction_label = bool(prediction)

        # Salvar a previsão no banco de dados
        Prediction.create(
            observation_id=observation_id,
            observation=json.dumps(data),
            proba=probability,
            true_class=None
        )

        # Construct response dictionary
        response = {
            "id": observation_id,
            "outcome": prediction_label,
        }
        return jsonify(response), 200
    except Exception as e:
        response["error"] = str(e)
        return jsonify(response), 500

@app.route('/recidivism_result/', methods=['POST'])
def recidivism_result():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['outcome']
        p.save()
        response = {
            "id": p.observation_id,
            "outcome": p.true_class,
            "predicted_outcome": p.proba >= 0.6
        }
        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = f'Observation ID: {obs["id"]} does not exist'
        return jsonify({'error': error_msg})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
