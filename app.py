from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__, template_folder='src/templates')

app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Fetching form data and initializing CustomData
        data = CustomData(
            quarter=request.form.get('quarter'),
            month=request.form.get('month'),
            day_of_month=request.form.get('day_of_month'),
            day_of_week=request.form.get('day_of_week'),
            op_unique_carrier=request.form.get('op_unique_carrier'),
            origin=request.form.get('origin'),
            dest=request.form.get('dest'),
            crs_dep_time=int(request.form.get('crs_dep_time'))  # Make sure this is an integer
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)
        print("Debug: Results from model prediction:", results)

        if results is not None:
            print("Debug: Sending prediction to template:", results[0])
            return render_template('home.html', prediction=results[0])
        else:
            print("Debug: Prediction failed or returned None")
            return render_template('home.html', error="Prediction failed")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
