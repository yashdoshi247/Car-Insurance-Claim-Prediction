from flask import Flask, render_template, request, Response, url_for
import pandas as pd

from src.pipeline.predicition_pipeline import PredictPipeline

app = Flask(__name__)

# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for predict data page
@app.route('/process_prediction', methods=['GET', 'POST'])
def process_prediction():
    if request.method == 'POST':
        # Get file from input form
        file = request.files['input_file']

        # Read file as pandas dataframe
        test_df = pd.read_csv(file)

        #Passing the test_df through the prediction pipeline
        pred_pipeline = PredictPipeline(test_df)
        predictions = pred_pipeline.predict()
        submission_df = pred_pipeline.make_submission_df(predictions)

        # Convert predictions to CSV file
        output = submission_df.to_csv(index=False)

        download_link = f'<p>Download your predictions file <a href="{url_for("download_file", filename="output.csv")}">here</a>.</p>'
        return render_template("predict.html", download_link=download_link)
    return render_template("predict.html")

@app.route("/download_file/<filename>")
def download_file(filename):
    return Response(
        filename,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=predictions.csv"})

if __name__ == '__main__':
    app.run()
