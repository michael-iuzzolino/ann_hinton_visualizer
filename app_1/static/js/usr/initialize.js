"use strict";
var VIS_CONTAINER_HEIGHT = 450;
var VIS_CONTAINER_WIDTH = 1000;
var STYLE_NAME_BY_SEX = false;

var default_part_2_architecture = [12, 6];
var default_part_3_architecture = [12];

var default_num_experiments = 1;
var num_experiments = default_num_experiments;

var default_num_epochs = 20;
var num_epochs = default_num_epochs;
var default_train_size = 89;
var train_size = default_train_size;
var max_train_size = 104;

var part_2_architecture = default_part_2_architecture;
var part_3_architecture = default_part_3_architecture;



var computing_interval = undefined;


function visualize_data(data, part, clf_result) {

    d3.select("#main_vis_div").remove();

    var result_x = 20;
    var result_y = 20;

    var row_height = 30;
    var row_width = 30;
    var weight_data = data["weights"];
    var name_data = data["names"];

    var female_names = ["Angela", "Charlotte", "Christine", "Francesca", "Gina", "Jennifer", "Lucia", "Margaret", "Maria", "Penelope", "Sophia", "Victoria"];
    var male_names = ["Alfonso", "Andrew", "Arthur", "Charles", "Christopher", "Colin", "Emilio", "James", "Marco", "Pierro", "Roberto", "Tomaso"];

    if (part === "part_2") {
        VIS_CONTAINER_HEIGHT = 550;
        VIS_CONTAINER_WIDTH = 1000;
        var legend_x = VIS_CONTAINER_WIDTH*0.9;
        var legend_y = 50;
        var plot_title = "Person1 to Distributed Person1 Layer Weights";
        var current_architecture = part_2_architecture;
    }
    else if (part === "part_3") {
        VIS_CONTAINER_HEIGHT = (row_height * weight_data.length)*1.4 + 200;
        VIS_CONTAINER_WIDTH = 1400;
        var legend_x = VIS_CONTAINER_WIDTH*0.9;
        var legend_y = 50;
        var plot_title = "Input to First Hidden Layer Weights";
        var current_architecture = part_3_architecture;
    }

    var hidden_neurons = [];
    for (var i=0; i < weight_data.length; i++) {
        hidden_neurons.push(i+1);
    }

    var min_weight = d3.min(weight_data, function(d) {
        return d3.min(d);
    });

    var max_weight = d3.max(weight_data, function(d) {
        return d3.max(d);
    });

    var sizeScale = d3.scaleLinear()
        .domain([min_weight, max_weight])
        .range([-20, 20]);

    var main_vis_div = d3.select("body").append("div").attr("id", "main_vis_div");

    var main_vis_svg = main_vis_div.append("svg").attr("id", "main_vis_svg")
        .attr("height", VIS_CONTAINER_HEIGHT)
        .attr("width", VIS_CONTAINER_WIDTH);

    main_vis_svg.append("rect")
        .attr("height", VIS_CONTAINER_HEIGHT)
        .attr("width", VIS_CONTAINER_WIDTH)
        .style("fill", "white")
        .style("stroke", "black");


    // PLOT TITLE -----------------
    // ---------------------------------------------------------------
    var plot_title_g = main_vis_svg.append("g")
        .attr("id", "plot_title_g")
        .attr("transform", "translate(300, 50)");

    plot_title_g.append("text")
        .attr("x", 5)
        .attr("y", 5)
        .text(plot_title)
        .style("font-size", "24px");
    // ---------------------------------------------------------------

    // RESULTS -----------------
    // ---------------------------------------------------------------
    var result_g = main_vis_svg.append("g")
        .attr("id", "result_g")
        .attr("transform", "translate("+result_x+", "+result_y+")");

    result_g.append("text")
        .attr("x", 5)
        .attr("y", 5)
        .text("Results");

    var mean_accuracy = clf_result["mean"];
    var std_accuracy = clf_result["std"];
    var num_experiments = clf_result["experiments"];
    var num_training_epochs = clf_result["epochs"];

    result_g.append("text")
        .attr("x", 5)
        .attr("y", 20)
        .text(function() {
            return "Hidden Architecture: [" + current_architecture + "]";
        })
        .style("font-size", "10px");

    result_g.append("text")
        .attr("x", 5)
        .attr("y", 35)
        .text(function() {
            return "Training Epochs: " + num_training_epochs;
        })
        .style("font-size", "10px");

    result_g.append("text")
        .attr("x", 5)
        .attr("y", 50)
        .text(function() {
            return "Num Experiments: " + num_experiments;
        })
        .style("font-size", "10px");

    result_g.append("text")
        .attr("x", 5)
        .attr("y", 65)
        .text(function() {
            return "Test Accuracy, MEAN: " + mean_accuracy.toFixed(2) + "%";
        })
        .style("font-size", "10px");

    result_g.append("text")
        .attr("x", 5)
        .attr("y", 80)
        .text(function() {
            return "Test Accuracy, STD: " + (std_accuracy / 100.0).toFixed(4);
        })
        .style("font-size", "10px");
    // ---------------------------------------------------------------

    // WEIGHTS ----------------
    // ---------------------------------------------------------------
    var tool_tip = d3.tip()
      .attr("class", "d3-tip")
      .offset([-8, 0])
      .html(function(d) { return "Weight: " + d; });

    main_vis_svg.call(tool_tip);

    var weight_group = main_vis_svg.append("g")
        .attr("transform", "translate(100, 200)");

    var name_group = weight_group.selectAll('g')
        .data(weight_data)
        .enter()
        .append('g')
        .attr('transform', function(d, i) {
            return 'translate(0, ' + (row_height + 5) * i + ')';
        });

    name_group.selectAll('rect.weights')
        .data(function(d) { return d; })
        .enter()
        .append('rect')
            .attr("class", "weights")
            .attr('x', function(d, i) {
                return (row_width + 5) * i;
            })
            .attr('width', function(d, i) {
                var size = Math.abs(sizeScale(d));
                return size;
            })
            .attr('height', function(d, i) {
                var size = Math.abs(sizeScale(d));
                return size;
            })
            .style('fill', function(d, i) {
                // return (sizeScale(d) >= 0) ? "red" : "blue";
                return (sizeScale(d) >= 0) ? "#0877bd" : "#f59322";
            })
            .style('stroke', "black")
            .style("opacity", 0.85)
            .on('mouseover', tool_tip.show)
            .on('mouseout', tool_tip.hide)

    weight_group.selectAll('text.names')
        .data(name_data).enter()
        .append("text")
        .attr("class", "names")
        .attr("x", 20)
        .attr("y", function(d, i) {
            return (row_width + 5) * i + 10;
        })
        .attr("transform", "rotate(-90)")
        .text(function(d, i) {
            return d;
        })
        .style("fill", function(d, i) {
            if (STYLE_NAME_BY_SEX) {
                for (var i=0; i < female_names.length; i++) {
                    if (d === female_names[i]) {
                        return "red";
                    }
                }
                return "blue";
            }
            else {
                return "black";
            }
        });

    weight_group.selectAll('text.neurons')
        .data(hidden_neurons).enter()
        .append("text")
        .attr("class", "neurons")
        .attr("x", -30)
        .attr("y", function(d, i) {
            return (row_width + 5) * i + 10;
        })
        .text(function(d, i) {
            return d;
        });

    weight_group.append("text")
        .attr("class", "neuron_label")
        .attr("x", -160)
        .attr("y", -50)
        .attr("transform", "rotate(-90)")
        .text("Hidden Neuron Number")

    // ---------------------------------------------------------------




    // LEGEND ----------------------
    // ---------------------------------------------------------------
    var legend_g = main_vis_svg.append("g")
        .attr("id", "legend_g")
        .attr("transform", "translate("+legend_x+", "+legend_y+")");

    legend_g.append("text")
        .attr("x", 5)
        .attr("y", 5)
        .text("Legend")

    var red_g = legend_g.append("g").attr("id", "red_g")
        .attr("transform", "translate(5, 15)");

    red_g.append("rect")
        .attr("height", 10)
        .attr("width", 10)
        .style("fill", "#0877bd")
        .style("stroke", "black")
        .style("opacity", 0.85);

    red_g.append("text")
        .attr("x", 15)
        .attr("y", 10)
        .text("+ weights")
        .style("font-size", "10px");


    var blue_g = legend_g.append("g").attr("id", "blue_g")
        .attr("transform", "translate(5, 35)");

    blue_g.append("rect")
        .attr("height", 10)
        .attr("width", 10)
        .style("fill", "#f59322")
        .style("stroke", "black")
        .style("opacity", 0.85);

    blue_g.append("text")
        .attr("x", 15)
        .attr("y", 10)
        .text("- weights")
        .style("font-size", "10px");
    // ---------------------------------------------------------------
}

function get_data(path_to_csv, part, clf_result) {
    $.ajax({
        url             :   "/read_data",
        method          :   'POST',
        contentType     :   'application/json',
        dataType        :   'json',
        data            :   JSON.stringify({"path_to_csv" : path_to_csv}),
        success : function(result) {
            console.log("HERE");
            d3.select("#training_tooltip_text").html("");
            d3.select("body").transition().duration(1000).style("opacity", 1.0).style("background-color", "white");
            clearInterval(computing_interval);
            visualize_data(result, part, clf_result);
        }
    });
}
function run_part_2_network() {
    $.ajax({
        url             :   "/run_part_2_network",
        method          :   'POST',
        contentType     :   'application/json',
        dataType        :   'json',
        data            :   JSON.stringify({"architecture" : part_2_architecture, "experiments" : num_experiments, "num_epochs" : num_epochs, "train_size" : train_size}),
        success : function(clf_result) {
            d3.select("#part_2_button").attr("value", "Generate Network - Part 2");
            var path_to_csv = "app_1/static/data/weight_data.csv";
            get_data(path_to_csv, "part_2", clf_result);
        }
    });
}

function run_part_3_network() {
    $.ajax({
        url             :   "/run_part_3_network",
        method          :   'POST',
        contentType     :   'application/json',
        dataType        :   'json',
        data            :   JSON.stringify({"architecture" : part_2_architecture, "experiments" : num_experiments, "num_epochs" : num_epochs, "train_size" : train_size}),
        success : function(clf_result) {
            d3.select("#part_3_button").attr("value", "Generate Network - Part 3");
            var path_to_csv = "app_1/static/data/part_3_weights.csv";
            get_data(path_to_csv, "part_3", clf_result);
        }
    });
}

function process_architecture(unprocess_architecture, part) {
    var split_arch = unprocess_architecture.split(",");

    if (part === "part_2") {
        part_2_architecture = [];
        for (var i=0; i < split_arch.length; i++) {
            part_2_architecture.push(parseInt(split_arch[i]));
        }
    }
    else if (part === "part_3") {
        part_3_architecture = [];
        for (var i=0; i < split_arch.length; i++) {
            part_3_architecture.push(parseInt(split_arch[i]));
        }
    }
}

function animate_training() {

    d3.select("#main_vis_svg").transition().duration(500).style("opacity", 0.0);

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
$(function() {

    d3.select("body").append("h1").attr("id", "main_title").html("ANN Weight Visualizer");

    var controls_div = d3.select('body').append("div").attr("id", "controls_div");
    var part_2_div = controls_div.append("div").attr("id", "part_2_div");
    part_2_div.append("input")
        .attr("id", "part_2_button")
        .attr("type", "button")
        .attr("value", "Generate Network - Part 2")
        .on("click", function() {
            d3.select(this).attr("value", "Training Network...");
            animate_training();
            run_part_2_network();
        });

    var part_2_options_div = part_2_div.append("div").attr("id", "part_2_options_div");
    part_2_options_div.append("label").html("Hidden Architecture: ");
    part_2_options_div.append('input')
        .attr('type','text')
        .attr('name','textInput')
        .attr('value', default_part_2_architecture)
        .on("change", function() {
            process_architecture(this.value, "part_2");
        });


    var part_3_div = controls_div.append("div").attr("id", "part_3_div");
    part_3_div.append("input")
        .attr("id", "part_3_button")
        .attr("type", "button")
        .attr("value", "Generate Network - Part 3")
        .on("click", function() {
            d3.select(this).attr("value", "Training Network...");
            animate_training();
            run_part_3_network();
        });

    var part_3_options_div = part_3_div.append("div").attr("id", "part_3_options_div");
    part_3_options_div.append("label").html("Hidden Architecture: ");
    part_3_options_div.append('input')
        .attr('type','text')
        .attr('name','textInput')
        .attr('value', default_part_3_architecture)
        .on("change", function() {
            process_architecture(this.value, "part_3");
        });

    var networking_training_tooltip_div = d3.select("body").append("div")
        .attr("id", "networking_training_tooltip_div")
        .append("h1")
        .attr("id", "training_tooltip_text");


    var experiment_div = controls_div.append("div").attr("id", "experiment_num_div");
    experiment_div.append("label").html("Experiments: ");
    experiment_div.append('input')
        .attr('type','text')
        .attr('name','textInput')
        .attr('value', default_num_experiments)
        .on("change", function() {
            num_experiments = this.value;
        });

    var epochs_div = controls_div.append("div").attr("id", "epochs_div");
    epochs_div.append("label").html("Training Epochs: ");
    epochs_div.append('input')
        .attr('type','text')
        .attr('name','textInput')
        .attr('value', default_num_epochs)
        .on("change", function() {
            num_epochs = this.value;
        });

    var train_size_div = controls_div.append("div").attr("id", "train_size_div");
    train_size_div.append("label").html("Training Size (max 104): ");
    train_size_div.append('input')
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



});
