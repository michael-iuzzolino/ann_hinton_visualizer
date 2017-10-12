from flask import render_template, jsonify, request
from app_1 import app
import numpy as np
import os
import pandas as pd
import app_1.python.main as dnn_main

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_network', methods=['GET', 'POST'])
def run_network():
    """
    var clf_data = {
        "architecture" : current_hidden_architecture,
        "experiments" : num_experiments,
        "num_epochs" : num_epochs,
        "train_size" : train_size,
        "dropout" : dropout,
        "activation_function" : activation_function,
        "loss_function" : loss_function,
        "dropout_keep_prob" : dropout_keep_prob,
        "architecture_type" : selected_architecture_type
    };
    """
    clf_params = {
        "architecture" : request.json['architecture'],
        "experiments" : int(request.json['experiments']),
        "n_epochs" : int(request.json['num_epochs']),
        "train_size" : int(request.json['train_size']),
        "dropout" : request.json['dropout'],
        "activation_function" : request.json['activation_function'],
        "loss_function" : request.json['loss_function'],
        "dropout_keep_prob" : request.json['dropout_keep_prob'],
    }

    architecture_type = request.json['architecture_type']

    df, clf, scoring_metrics = dnn_main.main(architecture_type, clf_params)

    names = list(df.index)
    weights = np.array(df.ix[:,:]).T.tolist()

    accuracies = {"training" : clf._metrics["training_accuracies"], "test" : clf._metrics["test_accuracies"]}

    return jsonify({"names_and_weights" : {"names" : names, "weights" : weights}, "scoring_metrics" : scoring_metrics, "accuracies" : accuracies})
