var STYLE_NAME_BY_SEX = false;


function visualizer_architecture() {

    d3.select("#net_arch_vis_div").remove();

    if (selected_architecture_type === "Hinton") {
        console.log(current_hidden_architecture);
    }
    else if (selected_architecture_type === "Fully Connected") {
        console.log(current_hidden_architecture);
    }

    var full_architecture = [2];
    for (var i=0; i < current_hidden_architecture.length; i++) {
        full_architecture.push(current_hidden_architecture[i]);
    }
    full_architecture.push(24);

    console.log(full_architecture);

    var container_height = 200;
    var container_width = 800;


    var row_height = 20;
    var row_width = 20;

    var input_r = 8;
    var default_r = 2;


    var net_arch_vis_div = d3.select("body").append("div").attr("id", "net_arch_vis_div");

    var net_arch_vis_svg = net_arch_vis_div.append("svg").attr("id", "net_arch_vis_svg")
        .attr("height", container_height)
        .attr("width", container_width);

    net_arch_vis_svg.append("rect")
        .attr("height", container_height)
        .attr("width", container_width)
        .style("fill", "white")
        .style("stroke", "black");

    // Header
    net_arch_vis_svg.append("text")
        .attr("x", function() { return container_width/2.0 - 50; })
        .attr("y", 20)
        .text("ANN Architecture");

    var plot_g = net_arch_vis_svg.append('g')
        .attr("id", "plot_g")
        .attr("transform", "rotate(-90)translate("+(-container_height+row_height)+","+row_width+")");

    var layer_group = plot_g.selectAll('g')
        .data(full_architecture)
        .enter()
        .append('g')
        .attr('transform', function(d, i) {
            var neuron_r = (i === 0) ? input_r : default_r;

            var unit_width = (container_width - (d*neuron_r + (d-1)*row_width)) / 2.0;
            var new_x = (row_width) * i;
            var new_y = unit_width;


            return 'translate('+new_x+', '+new_y+')';
        });


    layer_group.selectAll('circle.weights')
        .data(function(d, i) {
            var layer = [];
            for (var j=0; j < d; j++) {
                var val = (i === 0) ? input_r : default_r;
                layer.push(val);
            }
            return layer;
        })
        .enter()
        .append('circle')
            .attr("class", "weights")
            .attr('cy', function(d, i) {
                return (row_height) * i;
            })
            .attr('r', function(d, i) {
                return d;
            })
            .style('fill', "white")
            .style('stroke', "black")
            .style("opacity", 0.85);


}


function Hinton_Diagram_vis(data, clf_result) {

    d3.select("#hinton_vis_div").remove();

    var result_x = 20;
    var result_y = 20;

    var row_height = 30;
    var row_width = 30;
    var weight_data = data["weights"];
    var name_data = data["names"];

    var female_names = ["Angela", "Charlotte", "Christine", "Francesca", "Gina", "Jennifer", "Lucia", "Margaret", "Maria", "Penelope", "Sophia", "Victoria"];
    var male_names = ["Alfonso", "Andrew", "Arthur", "Charles", "Christopher", "Colin", "Emilio", "James", "Marco", "Pierro", "Roberto", "Tomaso"];

    if (selected_architecture_type === "Hinton") {
        VIS_CONTAINER_HEIGHT = 650;
        VIS_CONTAINER_WIDTH = 1000;
        var legend_x = VIS_CONTAINER_WIDTH*0.9;
        var legend_y = 50;
        var plot_title = "Person1 to Distributed Person1 Layer Weights";
        var current_architecture = architectures["Hinton"];
    }
    else if (selected_architecture_type === "Fully Connected") {
        VIS_CONTAINER_HEIGHT = (row_height * weight_data.length)*1.4 + 300;
        VIS_CONTAINER_WIDTH = 1400;
        var legend_x = VIS_CONTAINER_WIDTH*0.9;
        var legend_y = 50;
        var plot_title = "Input to First Hidden Layer Weights";
        var current_architecture = architectures["Fully Connected"];
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

    var hinton_vis_div = d3.select('#main_vis_div').append("div").attr("id", "hinton_vis_div");

    var main_vis_svg = hinton_vis_div.append("svg").attr("id", "main_vis_svg")
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

function train_test_plot(accuracies) {
    d3.select("#plot_vis_div").remove();
    var training_acc = accuracies["training"];
    var test_acc = accuracies["test"];

    console.log(accuracies);
    var PLOT_CONTAINER_HEIGHT = 450;
    var PLOT_CONTAINER_WIDTH = 450;

    var vis_height = 350;
    var vis_width = 350;

    var margin_top = (PLOT_CONTAINER_HEIGHT - vis_height) / 2;
    var margin_left = (PLOT_CONTAINER_WIDTH - vis_width) / 2;

    var circle_size = 2;


    var xScale = d3.scaleLinear()
        .domain([0, training_acc.length])
        .range([0, vis_width]);

    var xTestScale = d3.scaleLinear()
        .domain([0, test_acc.length])
        .range([0, vis_width]);

    var yScale = d3.scaleLinear()
        .domain([0, 100])
        .range([vis_height, 0]);

    var plot_vis_div = d3.select("#main_vis_div").append("div").attr("id", "plot_vis_div")
    plot_vis_svg = plot_vis_div.append("svg").attr("id", "plot_vis_svg")
        .attr("height", PLOT_CONTAINER_HEIGHT)
        .attr("width", PLOT_CONTAINER_WIDTH);

    plot_vis_svg.append("rect")
        .attr("height", PLOT_CONTAINER_HEIGHT)
        .attr("width", PLOT_CONTAINER_WIDTH)
        .style("fill", "white")
        .style("stroke", "black");


    var plot_g = plot_vis_svg.append('g')
        .attr("id", "plot_g")
        .attr("transform", "translate("+margin_left*1.1+", "+margin_top/2.0+")");

    plot_g.selectAll("circle.training")
        .data(training_acc).enter()
        .append("circle")
        .attr("class", "training")
        .attr('cx', function (d, i) {
            return xScale(i);
        })
        .attr('cy', function (d, i) {
            return yScale(d);
        })
        .attr('r', circle_size)
        .style("fill", "red")
        .style("stroke", "black");

    plot_g.selectAll("circle.test")
        .data(test_acc).enter()
        .append("circle")
        .attr("class", "test")
        .attr('cx', function (d, i) {
            return xTestScale(i);
        })
        .attr('cy', function (d, i) {
            return yScale(d);
        })
        .attr('r', circle_size)
        .style("fill", "blue")
        .style("stroke", "black");

    // Add the x Axis
    plot_g.append("g")
        .attr("transform", "translate(0," + vis_height + ")")
        .call(d3.axisBottom(xScale));

    // text label for the x axis
    plot_g.append("text")
        .attr("transform", "translate(" + (vis_width/2) + " ," + (vis_height + margin_top) + ")")
        .style("text-anchor", "middle")
        .text("Epochs");


    plot_g.append("g")
      .call(d3.axisLeft(yScale));

    // text label for the y axis
    plot_g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin_top)
      .attr("x", 0 - (vis_height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Accuracy %");

}

function visualize_data(data, clf_result, accuracies) {
    d3.select("#main_vis_div").remove();

    var main_vis_div = d3.select("#app_div").append("div").attr("id", "main_vis_div");

    Hinton_Diagram_vis(data, clf_result);

    train_test_plot(accuracies);
}
