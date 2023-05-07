from flask import Flask,request,render_template,jsonify
from main_folder.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    return "Success"


if __name__=="__main__":
    app.run(host='localhost',port=3000,debug=True)

