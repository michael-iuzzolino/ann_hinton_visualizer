"use strict";


// NETWORK ARCH PARAMS
// --------------------------------------------------------------------------------------------------
var architectures = {"Hinton" : [12, 6], "Fully Connected" : [12]};
var selected_architecture_type = "Hinton";
var current_hidden_architecture = architectures["Hinton"];

var loss_functions = {"Cross Entropy (Xentropy)" : "xentropy", "Mean Squared Error (MSE)" : "mse"};
var loss_function = loss_functions["Cross Entropy (Xentropy)"];

var activation_functions = {"ReLU" : "relu", "Sigmoid" : "sigmoid"};
var activation_function = activation_functions["ReLU"];

var dropout_default = false;
var dropout = dropout_default;

var dropout_keep_prob = 0.5;
// --------------------------------------------------------------------------------------------------

// EXPERIMENT / TRAINING PARAMS
// --------------------------------------------------------------------------------------------------
var default_num_experiments = 1;
var num_experiments = default_num_experiments;

var default_num_epochs = 500;
var num_epochs = default_num_epochs;
var default_train_size = 89;
var train_size = default_train_size;
var max_train_size = 104;
// --------------------------------------------------------------------------------------------------


var computing_interval = undefined;
var generating_network = false;
var network_vis_active = true;


function run_network() {

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

    $.ajax({
        url             :   "/run_network",
        method          :   'POST',
        contentType     :   'application/json',
        dataType        :   'json',
        data            :   JSON.stringify(clf_data),
        success : function(clf_result) {
            d3.select("#run_network_button").attr("value", "Generate Network");
            d3.select("#training_tooltip_text").html("");
            d3.select("body").transition().duration(1000).style("opacity", 1.0).style("background-color", "white");
            clearInterval(computing_interval);

            var names_and_weights = clf_result["names_and_weights"];
            var scoring_metrics = clf_result["scoring_metrics"];
            var accuracies = clf_result["accuracies"];

            generating_network = false;
            visualize_data(names_and_weights, scoring_metrics, accuracies);
        }
    });
}



function process_architecture(unprocess_architecture) {

    try {
        var split_arch = unprocess_architecture.split(",");
    }
    catch(err) {
        var split_arch = unprocess_architecture;
    }

    current_hidden_architecture = [];
    if (selected_architecture_type === "Hinton") {
        for (var i=0; i < split_arch.length; i++) {
            current_hidden_architecture.push(parseInt(split_arch[i]));
        }
    }
    else if (selected_architecture_type === "Fully Connected") {
        for (var i=0; i < split_arch.length; i++) {
            current_hidden_architecture.push(parseInt(split_arch[i]));
        }
    }

    visualizer_architecture();
}

function animate_training() {

    d3.select("#main_vis_div").transition().duration(500).style("opacity", 0.0);

    var phrase = "Training Network...".split("");

    var j=0;
    computing_interval = setInterval(function() {
        var text_arr = phrase.slice(0, j%phrase.length+1);
        var text_str = "";
        for (var k=0; k < text_arr.length; k++) {
            text_str += text_arr[k];
        }
        d3.select("#training_tooltip_text").html(text_str);
        j++;
    }, 100);
}

/**---------------------------------------------------------------------------------------------------------
 * MAIN
 *----------------------------------------------------------------------------------------------------------*/
function setup_network_params() {
    var network_parameters_main_div = d3.select("#controls_div").append('div').attr('id', 'network_parameters_main_div');


    // HEADER
    // ---------------------------------------------------------------
    network_parameters_main_div.append('h2').html("Network Parameters");
    // ---------------------------------------------------------------

    // ARCH TYPE DROPDOWN
    // ---------------------------------------------------------------
    var nn_type_div = network_parameters_main_div.append("div").attr("id", "nn_type_div");

    var nn_type_dropdown_form = nn_type_div.append("label")
        .html("Architecture Type")
        .append("form")
        .attr("id", "nn_type_dropdown_form")
        .append("select")
        .on("change", function() {
            selected_architecture_type = this.value;
            var hidden_arch = architectures[selected_architecture_type];
            d3.select("#hidden_architecture_field").attr("value", hidden_arch);
            process_architecture(hidden_arch);
        });

    nn_type_dropdown_form.selectAll("option")
        .data(Object.keys(architectures)).enter()
        .append("option")
        .attr("value", function(d, i) {
            return d;
        })
        .text(function(d, i) {
            return d;
        });
    // ---------------------------------------------------------------


    var network_parameters_div = network_parameters_main_div.append('div').attr('id', 'network_parameters_div');


    // HIDDEN ARCHITECTURE TEXTFIELD
    // ---------------------------------------------------------------
    var hidden_arch_div = network_parameters_div.append("div").attr("id", "hidden_arch_div");

    hidden_arch_div.append("label").html("Hidden Architecture: ");

    hidden_arch_div.append('input')
        .attr('id', 'hidden_architecture_field')
        .attr('type','text')
        .attr('name','textInput')
        .attr('value', architectures[selected_architecture_type])
        .on("change", function() {
            process_architecture(this.value);
        });
    // ---------------------------------------------------------------

    // LOSS FUNCTION DROPDOWN
    // ---------------------------------------------------------------
    var loss_function_div = network_parameters_div.append("div").attr("id", "loss_function_div");

    loss_function_div.append("label").html("Loss Function");

    var loss_function_form = loss_function_div.append("form")
        .attr("id", "loss_function_form");

    var loss_function_select = loss_function_form.append("select")
        .on("change", function() {
            loss_function = loss_functions[this.value];
        });

    loss_function_select.selectAll("option")
        .data(Object.keys(loss_functions)).enter()
        .append("option")
        .attr("value", function(d, i) {
            return d;
        })
        .text(function(d, i) {
            return d;
        });
    // ---------------------------------------------------------------


    // ACTIVATION FUNCTION DROPDOWN
    // ---------------------------------------------------------------
    var activation_function_div = network_parameters_div.append("div").attr("id", "activation_function_div");

    activation_function_div.append("label").html("Activation Function");

    var activation_function_form = activation_function_div.append("form")
        .attr("id", "activation_function_form");

    var activation_function_select = activation_function_form.append("select")
        .on("change", function() {
            activation_function = activation_functions[this.value];
        });

    activation_function_select.selectAll("option")
        .data(Object.keys(activation_functions)).enter()
        .append("option")
        .attr("value", function(d, i) {
            return d;
        })
        .text(function(d, i) {
            return d;
        });
    // ---------------------------------------------------------------

    // DROPOUT
    // ---------------------------------------------------------------
    var dropout_div = network_parameters_div.append("div").attr("id", "dropout_div");

    dropout_div.append("label").html("Dropout");

    var dropout_form = dropout_div.append("form")
        .attr("id", "dropout_form");

    var dropout_select = dropout_form.append("select")
        .on("change", function() {
            dropout = (this.value == "On") ? true : false;
        });

    dropout_select.selectAll("option")
        .data(["Off", "On"]).enter()
        .append("option")
        .attr("value", function(d, i) {
            return d;
        })
        .text(function(d, i) {
            return d;
        });

    var dropout_text_div = dropout_div.append("div").attr("id", "dropout_text_div");

    dropout_text_div.append("label").html("Keep Probability: ");

    var slider_val_div = dropout_text_div.append("div").attr("id", "slider_val_div");
    slider_val_div.append("p").attr('id', "dropout_keep_prob_slider_text").html(function() { return dropout_keep_prob.toFixed(2) });

    var slider_div = dropout_div.append("div").attr("id", "slider_div");
    slider_div.append('input')
        .attr("id", "keep_prob_slider")
        .attr("type", "range")
        .attr("value", 50)
        .attr("min", 0)
        .attr("max", 100)
        .attr("step", 5)
        .on("change", function() {
            dropout_keep_prob = this.value / 100.0;
            d3.select('#dropout_keep_prob_slider_text').html(function() { return dropout_keep_prob.toFixed(2); });
        });
    // ---------------------------------------------------------------

    // Generate Network Button
    // ------------------------------------------------------------
    var view_net_button_div = network_parameters_main_div.append('div').attr("id", "view_net_button_div");

    view_net_button_div.append("input")
        .attr("id", "view_net_button")
        .attr("type", "button")
        .attr("value", "Hide Network")
        .on("click", function() {
            toggle_net_vis();
            network_vis_active = !network_vis_active;
        });
    // ------------------------------------------------------------
}

function toggle_net_vis(hide) {

    if (hide !== undefined) {
        d3.select("#net_arch_vis_div").transition().duration(1000).style("opacity", 0.0);
        d3.select('#view_net_button').attr("value", "Show Network");
        network_vis_active = false;
        return;
    }

    if (network_vis_active) {
        d3.select("#net_arch_vis_div").transition().duration(1000).style("opacity", 0.0);
        d3.select('#view_net_button').attr("value", "Show Network");
    }
    else {
        d3.select("#net_arch_vis_div").transition().duration(1000).style("opacity", 1.0);
        d3.select('#view_net_button').attr("value", "Hide Network");
    }
}

function setup_training_params() {
    var training_params_div = d3.select("#controls_div").append("div").attr("id", "training_params_div");

    // HEADER
    // ---------------------------------------------------------------
    training_params_div.append('h2').html("Training Parameters");
    // ---------------------------------------------------------------

    // NUM EXPERIMENTS TEXTFIELD
    // ---------------------------------------------------------------
    var experiments_div = training_params_div.append("div").attr("id", "experiments_div");
    experiments_div.append("label").html("Experiments: ");
    experiments_div.append('input')
        .attr('id','experiments_text_field')
        .attr('type','text')
        .attr('name','textInput')
        .attr('value', default_num_experiments)
        .on("change", function() {
            num_experiments = this.value;
        });
    // ---------------------------------------------------------------

    // TRAINING EPOCHS TEXTFIELD
    // ---------------------------------------------------------------
    var epochs_div = training_params_div.append("div").attr("id", "epochs_div");
    epochs_div.append("label").html("Training Epochs: ");
    epochs_div.append('input')
        .attr('id','training_epochs_text_field')
        .attr('type','text')
        .attr('name','textInput')
        .attr('value', default_num_epochs)
        .on("change", function() {
            num_epochs = this.value;
        });
    // ---------------------------------------------------------------

    // TRAINING SIZE TEXTFIELD
    // ---------------------------------------------------------------
    var train_size_div = training_params_div.append("div").attr("id", "train_size_div");
    train_size_div.append("label").html("Training Size (max 104): ");
    train_size_div.append('input')
        .attr('id','training_size_text_field')
        .attr('type','text')
        .attr('name','textInput')
        .attr('value', default_train_size)
        .on("change", function() {
            if (this.value > max_train_size) {
                train_size = max_train_size;
                this.value = max_train_size;
            }
            else {
                train_size = this.value;
            }
        });
    // ---------------------------------------------------------------
}

function controls_setup() {
    var controls_div = d3.select('#app_div').append("div").attr("id", "controls_div");

    // WEIGHT MOUSEOVER TOOLTIP
    // -------------------------------------------------------------------
    var networking_training_tooltip_div = d3.select("body").append("div")
        .attr("id", "networking_training_tooltip_div")
        .append("h1")
        .attr("id", "training_tooltip_text");
    // -------------------------------------------------------------------

    // NETWORK PARAMETERS
    // -----------------------
    setup_network_params();
    // -----------------------

    // Training Parameters
    // -----------------------
    setup_training_params();
    // -----------------------

    // Generate Network Button
    // ------------------------------------------------------------
    var run_network_button_div = controls_div.append('div').attr("id", "run_network_button_div");

    run_network_button_div.append("input")
        .attr("id", "run_network_button")
        .attr("type", "button")
        .attr("value", "Generate Network")
        .on("click", function() {
            if (generating_network) {
                return;
            }
            toggle_net_vis("hide");
            generating_network = true;
            d3.select(this).attr("value", "Training Network...");
            animate_training();
            run_network();
        });
    // ------------------------------------------------------------
}

$(function() {

    d3.select("body").append("h1").attr("id", "main_title").html("ANN Weight Visualizer");

    d3.select("body").append("div").attr("id", "app_div");

    controls_setup();

    visualizer_architecture();
});
