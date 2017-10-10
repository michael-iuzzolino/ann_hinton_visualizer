import numpy as np
import pandas as pd
import sklearn.utils as skutils
from app_1.python.neural_networks import RelationshipDNN, DNN
from app_1.python.helper import read_family_tree_data

def split_data(dataset, train_size):
    X = dataset["X"]
    y = dataset["y"]
    X_shuff, y_shuff = skutils.shuffle(X, y)

    num_people = dataset["names"].shape[0]
    num_relatioons = dataset["relationships"].shape[0]

    return {
        "training" : {
            "X_person_train"    :   X_shuff[:train_size, :num_people],
            "X_relation_train"  :   X_shuff[:train_size, num_people:],
            "y_train"           :   y_shuff[:train_size]
        },
        "test" : {
            "X_person_test"     :   X_shuff[train_size:, :num_people],
            "X_relation_test"   :   X_shuff[train_size:, num_people:],
            "y_test"            :   y_shuff[train_size:]
        }
    }

def split_data_part_3(dataset, train_size):
    X = dataset["X"]
    y = dataset["y"]
    X_shuff, y_shuff = skutils.shuffle(X, y)

    num_people = dataset["names"].shape[0]
    num_relatioons = dataset["relationships"].shape[0]

    return {
        "training" : {
            "X"    :   X_shuff[:train_size, :],
            "y"    :   y_shuff[:train_size]
        },
        "test" : {
            "X"     :   X_shuff[train_size:, :],
            "y"     :   y_shuff[train_size:]
        }
    }

def part2(architecture, experiments, n_epochs, train_size):
    relationship_dataset = read_family_tree_data()

    top_accuracies = []
    for i in range(int(experiments)):
        print("Experiment {}".format(i))
        data = split_data(relationship_dataset, train_size=train_size)
        training_data = data["training"]
        test_data = data["test"]

        clf = RelationshipDNN(hidden_layers=architecture, num_classes=24, n_epochs=n_epochs, activation_fxn="sigmoid", verbose=True)
        clf.fit(training_data, test_data)

        train_acc = clf._metrics["training_accuracies"][-1]
        test_acc = clf._metrics["test_accuracies"][-1]
        top_accuracies.append(test_acc)
        print("\n")

    test_acc_mean = np.mean(top_accuracies)
    test_acc_std = np.std(top_accuracies)


    person_1_layer_weights = clf.get_weights()[0][0]
    df = pd.DataFrame(person_1_layer_weights, index=relationship_dataset["names"], columns=[i for i in range(1, 7)])
    df.to_csv("app_1/static/data/weight_data.csv", sep=",")

    return clf, {"mean" : test_acc_mean, "std" : test_acc_std, "experiments" : experiments, "epochs" : n_epochs}


def part3(architecture, experiments, n_epochs, train_size):
    relationship_dataset = read_family_tree_data()
    num_classes = 24

    top_accuracies = []
    for i in range(int(experiments)):
        print("Experiment {}".format(i))
        data = split_data_part_3(relationship_dataset, train_size=train_size)
        training_data = data["training"]
        test_data = data["test"]

        clf = DNN(hidden_layers=architecture, num_classes=num_classes, loss_fxn="xentropy", batch_size=89, n_epochs=n_epochs, activation_fxn="sigmoid", verbose=True)
        clf.fit(training_data, test_data)

        train_acc = clf._metrics["training_accuracies"][-1]
        test_acc = clf._metrics["test_accuracies"][-1]
        top_accuracies.append(test_acc)
        print("\n")

    test_acc_mean = np.mean(top_accuracies)
    test_acc_std = np.std(top_accuracies)

    W1 = clf.get_weights()[0]
    df = pd.DataFrame(W1, index=np.r_[relationship_dataset["names"], relationship_dataset["relationships"]], columns=[i for i in range(1, architecture[0]+1)])
    df.to_csv("app_1/static/data/part_3_weights.csv", sep=",")

    return clf, {"mean" : test_acc_mean, "std" : test_acc_std, "experiments" : experiments, "epochs" : n_epochs}


def main(architecture, experiments, part, n_epochs=5000, train_size=89):
    if part is "part_2":
        clf, results = part2(architecture, experiments, n_epochs, train_size)
    elif part is "part_3":
        clf, results = part3(architecture, experiments, n_epochs, train_size)

    return clf, results

if __name__ == "__main__":
    main()
