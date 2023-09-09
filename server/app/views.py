from flask import Flask, request, jsonify
import numpy as np

#######################################################################################
from app import app
from common.pereferences import DEBUG, PORT, HOST, THREADED
from common.helpers import get_all_unique_data, get_all_stat, get_by_query

@app.route('/')
@app.route('/index')
def source():
    return "SERVER RUN..."

@app.route('/get_all_query')
def get_all_query():
    return {'questions': get_all_unique_data('query', data_path='../data/all_data.csv').tolist()}

@app.route('/get_dash_data')
def get_dash_data():
    return get_all_stat(data_path='../data/labeled_data.csv')

@app.route('/get_data_by_query', methods=['POST'])
def get_data_by_query():
    return get_by_query(request.form['query'],data_path='../data/labeled_data.csv')

def main():
    app.run(HOST, PORT, debug=DEBUG, threaded=THREADED)

if __name__ == '__main__':
    main()