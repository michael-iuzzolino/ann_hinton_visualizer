import pandas as pd
import numpy as np

def read_family_tree_data():

    # Get names and relationships
    names = []
    relationships = []
    raw_file_data = []
    with open("app_1/static/data/relations.txt", "r") as infile:
        for line in infile:
            raw_file_data.append(line)
            split_line = line.split()
            names.append(split_line[0])
            relationships.append(split_line[1])
            names.append(split_line[2])

    # Make names/relationships lists unique
    names = np.unique(names)
    relationships = np.unique(relationships)

    # One-hot encoding
    X, y = [], []
    for relationship in raw_file_data:
        split_relation = relationship.split()

        x_name = split_relation[0]
        x_relation = split_relation[1]
        y_name = split_relation[2]

        X_one_hot_names = np.zeros(len(names))
        X_one_hot_names[np.where(names == x_name)] = 1.0

        X_one_hot_relations = np.zeros(len(relationships))
        X_one_hot_relations[np.where(relationships == x_relation)] = 1.0

        X_data = np.r_[X_one_hot_names, X_one_hot_relations]

        Y_one_hot_names = np.zeros(len(names))
        Y_one_hot_names[np.where(names == y_name)] = 1.0

        X.append(X_data)
        y.append(Y_one_hot_names)

    return {"X" : np.array(X), "y" : np.array(y), "names" : names, "relationships" : relationships}
