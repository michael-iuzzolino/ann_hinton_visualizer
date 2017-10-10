import tensorflow as tf
from sklearn.utils import shuffle
import sys

class RelationshipDNN(object):
    def __init__(self, hidden_layers=[12, 6], num_classes=1,
        learning_rate=0.001, n_epochs=500,
        activation_fxn="sigmoid", test_probe_freq=0.05,
        visualize_the_weights=False, visualize_the_gradients=False,
        verbose=True):

        tf.reset_default_graph()

        self._verbose = verbose

        self._person_hidden_layer = 6
        self._relation_hidden_layer = 6
        self._hidden_layers = [12, 6]
        self._num_classes = num_classes

        self._alpha = learning_rate
        self._n_epochs = n_epochs
        self._threshold_value = 0.5

        self._test_probe_freq = test_probe_freq


        self._cbar_plotted = False

        self._activations = {"sigmoid" : tf.nn.sigmoid, "relu" : tf.nn.relu}
        self._activation_fxn = self._activations[activation_fxn]

        self._sess = tf.Session()

    def setup_loss(self, y_hat, y_train):
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y_train)
        return cross_entropy

    def setup_optimizer(self, loss):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(self._alpha)
        return optimizer

    def setup_performance_metric(self, y_hat, y):
        with tf.name_scope("performance_metric"):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def build_model(self, X_person, X_relation):

        self._weights = []
        self._biases = []
        self._z = []

        # Input layer
        with tf.variable_scope("person_input"):
            W_person_input = tf.get_variable("weights_input_person", shape=[X_person.get_shape()[1], self._person_hidden_layer], initializer=tf.contrib.layers.xavier_initializer())
            b_person_input = tf.get_variable("bias_input_person", shape=[self._person_hidden_layer], initializer=tf.contrib.layers.xavier_initializer())
            z_person_input = self._activation_fxn(tf.matmul(X_person, W_person_input) + b_person_input, name="activation_input_person")

        with tf.variable_scope("relation_input"):
            W_relation_input = tf.get_variable("weights_input_relation", shape=[X_relation.get_shape()[1], self._relation_hidden_layer], initializer=tf.contrib.layers.xavier_initializer())
            b_relation_input = tf.get_variable("bias_input_relation", shape=[self._relation_hidden_layer], initializer=tf.contrib.layers.xavier_initializer())
            z_relation_input = self._activation_fxn(tf.matmul(X_relation, W_relation_input) + b_relation_input, name="activation_input_relation")

        self._weights.append([W_person_input, W_relation_input])

        full_fan_out = self._person_hidden_layer + self._relation_hidden_layer
        combined_output = tf.concat([z_person_input, z_relation_input], axis=1)

        # Hidden layers
        prev_layer_output = combined_output
        layer_i = 0
        for layer_i in range(len(self._hidden_layers)-1):

            fan_in = self._hidden_layers[layer_i]
            fan_out = self._hidden_layers[layer_i+1]

            with tf.variable_scope("hidden_{}".format(layer_i+1)):
                W_hidden = tf.get_variable("weights_{}".format(layer_i+1), shape=[fan_in, fan_out], initializer=tf.contrib.layers.xavier_initializer())
                b_hidden = tf.get_variable("bias_{}".format(layer_i+1), shape=[fan_out], initializer=tf.contrib.layers.xavier_initializer())
                z_hidden = self._activation_fxn(tf.matmul(prev_layer_output, W_hidden) + b_hidden, name="activation_{}".format(layer_i+1))

            self._weights.append(W_hidden)

            prev_layer_output = z_hidden

        # Output layers
        with tf.variable_scope("output"):
            W_3 = tf.get_variable("w_out", shape=[self._hidden_layers[-1], self._num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b_3 = tf.get_variable("bias_out", shape=[self._num_classes], initializer=tf.contrib.layers.xavier_initializer())
            y_hat = tf.matmul(prev_layer_output, W_3) + b_3

        self._weights.append(W_3)

        return y_hat

    def fit(self, training_data, test_data):
        X_person_train = training_data["X_person_train"]
        X_relation_train = training_data["X_relation_train"]
        y_train = training_data["y_train"]

        X_person_test = test_data["X_person_test"]
        X_relation_test = test_data["X_relation_test"]
        y_test = test_data["y_test"]

        X_person = tf.placeholder(tf.float32, shape=[None, X_person_train.shape[1]])
        X_relation = tf.placeholder(tf.float32, shape=[None, X_relation_train.shape[1]])
        y = tf.placeholder(tf.float32, shape=[None, self._num_classes])

        y_hat = self.build_model(X_person, X_relation)
        loss = self.setup_loss(y_hat, y)


        accuracy = self.setup_performance_metric(y_hat, y)
        optimizer = self.setup_optimizer(loss)
        opt_min = optimizer.minimize(loss)
        grad_step = optimizer.compute_gradients(loss)

        # Initialize all variables
        self._sess.run(tf.global_variables_initializer())

        # Initialize metric containers
        self._metrics = {"losses" : [], "training_accuracies" : [], "test_accuracies" : []}

        for epoch in range(self._n_epochs):
            _, train_acc = self._sess.run([opt_min, accuracy], feed_dict={X_person: X_person_train, X_relation: X_relation_train, y: y_train})

            # Store metrics
            self._metrics["training_accuracies"].append(train_acc * 100)

            if epoch % int(self._n_epochs * self._test_probe_freq) == 0:

                # Get Test accuracy
                test_acc = self._sess.run(accuracy, feed_dict={X_person: X_person_test, X_relation: X_relation_test, y: y_test})

                # Store metrics
                self._metrics["test_accuracies"].append(test_acc * 100)

                if self._verbose:
                    sys.stdout.write("\rEpoch: {:5d} -- Train: {:5.2f}% -- Test: {:5.2f}%".format(epoch, train_acc*100, test_acc*100))
                    sys.stdout.flush()


        # Get Test accuracy
        test_acc = self._sess.run(accuracy, feed_dict={X_person: X_person_test, X_relation: X_relation_test, y: y_test})

        # Store metrics
        self._metrics["test_accuracies"].append(test_acc * 100)

    def get_weights(self):
        weights = []
        for i, w in enumerate(self._weights):
            if i == 0:
                W_person = self._sess.run(w[0])
                W_relation = self._sess.run(w[1])
                weights.append([W_person, W_relation])
            else:
                W = self._sess.run(w)
                weights.append(W)

        return weights



class DNN(object):
    def __init__(self, hidden_layers=[10], num_classes=1,
        loss_fxn="mse",
        learning_rate=0.001, n_epochs=500, batch_size=100,
        activation_fxn="sigmoid", test_probe_freq=0.1,
        visualize_the_weights=False, visualize_the_gradients=False,
        verbose=True):

        tf.reset_default_graph()

        self._verbose = verbose

        self._hidden_layers = hidden_layers
        self._num_classes = num_classes
        self._alpha = learning_rate
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._threshold_value = 0.5
        self._loss_fxn = loss_fxn

        self._test_probe_freq = test_probe_freq


        self._cbar_plotted = False

        self._activations = {"sigmoid" : tf.nn.sigmoid, "relu" : tf.nn.relu}

        self._activation_fxn = self._activations[activation_fxn]
        self._sess = tf.Session()

    def setup_loss(self, y_hat, y_train):
        with tf.name_scope("loss"):
            if self._loss_fxn is "mse":
                loss = tf.reduce_mean(tf.square(y_hat - y_train))
            elif self._loss_fxn is "xentropy":
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=y_hat))
        return loss

    def setup_optimizer(self, loss):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(self._alpha)
        return optimizer

    def setup_performance_metric(self, y_hat, y):
        with tf.name_scope("performance_metric"):

            if self._loss_fxn is "mse":
                y_hat_threshold = tf.cast(tf.cast(y_hat + self._threshold_value, tf.int32), tf.float32)
                correct_prediction = tf.equal(y_hat_threshold, y)
                performance = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            elif self._loss_fxn is "xentropy":
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
                performance = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return performance

    def get_next_batch(self, X, y):
        X_shuff, y_shuff = shuffle(X, y)
        return X_shuff[:self._batch_size], y_shuff[:self._batch_size]

    def build_model(self, X):

        self._weights = []
        self._biases = []
        self._z = []


        # Input layer
        with tf.variable_scope("input"):
            W_input = tf.get_variable("weights_input", shape=[X.get_shape()[1], self._hidden_layers[0]], initializer=tf.contrib.layers.xavier_initializer())
            b_input = tf.get_variable("bias_input", shape=[self._hidden_layers[0]], initializer=tf.contrib.layers.xavier_initializer())
            z_input = self._activation_fxn(tf.matmul(X, W_input) + b_input, name="activation_input")

            self._weights.append(W_input)
            self._biases.append(b_input)
            self._z.append(z_input)

        # Hidden layers
        prev_layer_output = z_input
        layer_i = 0
        for layer_i in range(len(self._hidden_layers)-1):

            fan_in = self._hidden_layers[layer_i]
            fan_out = self._hidden_layers[layer_i+1]

            with tf.variable_scope("hidden_{}".format(layer_i+1)):
                W_hidden = tf.get_variable("weights_{}".format(layer_i+1), shape=[fan_in, fan_out], initializer=tf.contrib.layers.xavier_initializer())
                b_hidden = tf.get_variable("bias_{}".format(layer_i+1), shape=[fan_out], initializer=tf.contrib.layers.xavier_initializer())
                z_hidden = self._activation_fxn(tf.matmul(prev_layer_output, W_hidden) + b_hidden, name="activation_{}".format(layer_i+1))

            self._weights.append(W_hidden)
            self._biases.append(b_hidden)
            self._z.append(z_hidden)

            prev_layer_output = z_hidden

        # Output layers
        with tf.variable_scope("output"):
            W = tf.get_variable("weights_{}".format(layer_i+1), shape=[self._hidden_layers[-1], self._num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("bias_{}".format(layer_i+1), shape=[self._num_classes], initializer=tf.contrib.layers.xavier_initializer())
            y_hat = tf.matmul(prev_layer_output, W) + b

            if self._num_classes == 1:
                y_hat = tf.nn.sigmoid(y_hat)

            self._weights.append(W)
            self._biases.append(b)
            self._z.append(y_hat)

        return self._z[-1]

    def fit(self, training_data, test_data):
        X_train = training_data["X"]
        y_train = training_data["y"]
        X_test = test_data["X"]
        y_test = test_data["y"]

        X = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])
        y = tf.placeholder(tf.float32, shape=[None, self._num_classes])

        y_hat = self.build_model(X)
        loss = self.setup_loss(y_hat, y)

        accuracy = self.setup_performance_metric(y_hat, y)
        optimizer = self.setup_optimizer(loss)
        opt_min = optimizer.minimize(loss)
        grad_step = optimizer.compute_gradients(loss)

        # Save variables for prediction
        self._X = X
        self._y = y
        self._y_hat = y_hat

        # Initialize all variables
        self._sess.run(tf.global_variables_initializer())

        # Initialize metric containers
        self._metrics = {"losses" : [], "training_accuracies" : [], "test_accuracies" : []}


        for epoch in range(self._n_epochs):
            X_batch, y_batch = self.get_next_batch(X_train, y_train)
            _, train_acc, loss_val = self._sess.run([opt_min, accuracy, loss], feed_dict={X: X_batch, y: y_batch})

            # Store metrics
            self._metrics["losses"].append(loss_val)
            self._metrics["training_accuracies"].append(train_acc * 100)

            if epoch % int(self._n_epochs * self._test_probe_freq) == 0:

                # Get Test accuracy
                test_acc = self._sess.run(accuracy, feed_dict={X: X_test, y: y_test})

                # Store metrics
                self._metrics["test_accuracies"].append(test_acc * 100)

                if self._verbose:
                    sys.stdout.write("\rEpoch: {:5d} -- Loss: {:5.2f} -- Train: {:5.2f}% -- Test: {:5.2f}%".format(epoch, loss_val, train_acc*100, test_acc*100))
                    sys.stdout.flush()

    def predict(self, test_point):
        prediction = tf.cast(self._y_hat + self._threshold_value, tf.int32)
        return self._sess.run(prediction, feed_dict={self._X: test_point})


    def get_weights(self):
        weights = []
        for i, w in enumerate(self._weights):
            W = self._sess.run(w)
            weights.append(W)

        return weights
