import json, analysis
from flask import Flask, request, jsonify, render_template,  redirect, url_for

app = Flask(__name__)


@app.route('/')
def index():
     return render_template('index.html')

@app.route('/api/analysis', methods = ['POST'])
def ner():
    handle = request.get_json().get('TwitterHandle', '')
    index = analysis.depression_analysis(handle)
    response = app.response_class(
        response = json.dumps(index),
        status = 200,
        mimetype = 'application/json'
    )
    return response



if __name__ == '__main__':
    app.run(debug = True)