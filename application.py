from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create the Flask application
application = Flask(__name__)

# Alias the application as 'app'
app = application

# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for the prediction form
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # For a GET request, pass an empty form or default values
        return render_template('home.html', results=None, form=request.form)
    else:
        # Gathering data from form
        data = CustomData(
            CRS_DEP_TIME=float(request.form.get('CRS_DEP_TIME')),
            MONTH=float(request.form.get('MONTH')),
            DAY_OF_WEEK=float(request.form.get('DAY_OF_WEEK')),
            OP_UNIQUE_CARRIER=request.form.get('OP_UNIQUE_CARRIER'),
            ORIGIN=request.form.get('ORIGIN'),
            DEST=request.form.get('DEST')
        )

        # Converting data to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Before Prediction")
        
        # Creating prediction pipeline instance and predicting
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")

        # Adjust how results are handled
        if isinstance(results, np.ndarray) and results.ndim > 0:
            result_to_display = results[0]
        else:
            result_to_display = results  # If results is a scalar, use it directly

        # Render prediction results
        return render_template('home.html', results=result_to_display, form=request.form)

# Running the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
