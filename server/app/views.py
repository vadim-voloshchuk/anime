from flask import Flask, request, jsonify, render_template
import numpy as np

#######################################################################################
from app import app
from common.pereferences import DEBUG, PORT, HOST, THREADED
from common.helpers import get_all_unique_data, get_all_stat, get_by_query, get_clusters

@app.route('/')
@app.route('/index')
def source():
    return render_template("index.html")

@app.route('/get_all_query')
def get_all_query():
    return {'questions': get_all_unique_data('query', data_path='../data/all_data.csv').tolist()}

@app.route('/testing_clustering')
def testing_clustering():
    return render_template("qr_testing.html")

@app.route('/ag_clustering', methods = ['POST'])
def ag_clustering():
    return render_template("result.html", route = "/testing_clustering", result = get_clusters(request.form['query'],data_path='../data/all_data_sifted_2.csv'))

@app.route('/get_dash_data')
def get_dash_data():
    return get_all_stat(data_path='../data/labeled_data.csv')

@app.route('/get_data_by_query', methods=['POST'])
def get_data_by_query():
    return get_by_query(request.form['query'],data_path='../data/labeled_data.csv')

@app.route('/get_clusters_by_query', methods=['POST'])
def get_clusters_by_query():
    return get_clusters(request.form['query'],data_path='../data/all_data_sifted_2.csv')

def main():
    app.run(HOST, PORT, debug=DEBUG, threaded=THREADED)

if __name__ == '__main__':
    main()