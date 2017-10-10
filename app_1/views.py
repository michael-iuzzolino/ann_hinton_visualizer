from flask import render_template, jsonify, request
from app_1 import app
import numpy as np
import os
import pandas as pd
import app_1.python.main as dnn_main

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_part_2_network', methods=['GET', 'POST'])
def run_part_2_network():
    architecture = request.json['architecture']
    experiments = int(request.json['experiments'])
    n_epochs = int(request.json['num_epochs'])
    train_size = int(request.json['train_size'])

    clf, results = dnn_main.main(architecture, experiments, "part_2", n_epochs=n_epochs, train_size=train_size)

    return jsonify(results)

@app.route('/run_part_3_network', methods=['GET', 'POST'])
def run_part_3_network():
    architecture = request.json['architecture']
    experiments = request.json['experiments']
    n_epochs = int(request.json['num_epochs'])
    train_size = int(request.json['train_size'])

    clf, results = dnn_main.main(architecture, experiments, "part_3", n_epochs=n_epochs, train_size=train_size)

    return jsonify(results)


@app.route('/read_data', methods=['GET', 'POST'])
def read_data():
    path_to_csv = request.json['path_to_csv']

    df = pd.read_csv(path_to_csv, sep=",")
    names = list(df.ix[:,0])
    weights = np.array(df.ix[:, 1:]).T.tolist()

    return jsonify({"names" : names, "weights" : weights})
