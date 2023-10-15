function sigmoid(x) 
{
    return 1 / (1 + Math.exp(-x))
}

function dsigmoid(y) {
    return y * (1-y)
}

function ReLU(x) {
    if (x <= 0) {
        return 0
    } else {
        return x
    }
}

function dReLU(y) {
    if (y > 0) {
        return 1
    } else if (y < 0) {
        return 0
    } else {
        return 0
    }
}

class NeuralNetwork 
{
    constructor(input_nodes, hidden_nodes, output_nodes) 
    {
        this.input_nodes = input_nodes
        this.hidden_nodes = hidden_nodes
        this.output_nodes= output_nodes

        this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes)
        this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes)
        this.weights_ih.randomize()
        this.weights_ho.randomize()

        this.bias_h = new Matrix(this.hidden_nodes, 1)
        this.bias_o = new Matrix(this.output_nodes, 1)
        this.bias_h.randomize()
        this.bias_o.randomize()

        this.learning_rate = 0.001
    }

    serialize() {
        return JSON.stringify(this);
      }
    
    static deserialize(data) {
        if (typeof data == 'string') {
            data = JSON.parse(data);
        }
        let nn = new NeuralNetwork(data.input_nodes, data.hidden_nodes, data.output_nodes);
        nn.weights_ih = Matrix.deserialize(data.weights_ih);
        nn.weights_ho = Matrix.deserialize(data.weights_ho);
        nn.bias_h = Matrix.deserialize(data.bias_h);
        nn.bias_o = Matrix.deserialize(data.bias_o);
        nn.learning_rate = data.learning_rate;
        return nn;
    }

    feedForward(input_array) 
    {
        // Generates hidden layer output
        let inputs = Matrix.fromArray(input_array)

        // Feeds input forward to the hidden layer, and applies activation function
        let hidden = Matrix.multiply(this.weights_ih, inputs)
        hidden.add(this.bias_h)
        hidden.map(sigmoid)

        // Fee
        let output = Matrix.multiply(this.weights_ho, hidden)
        output.add(this.bias_o)
        output.map(sigmoid)
        return output.toArray()
    }

    train(input_array, target_array)
    {
        /**
         * Feeds input forward through network
         */
        // Converts input array into vector
        let inputs = Matrix.fromArray(input_array)

        // Feeds input forward to the hidden layer, and applies activation function
        let hidden = Matrix.multiply(this.weights_ih, inputs)
        hidden.add(this.bias_h)
        hidden.map(sigmoid)

        // Feeds hidden layer values as input to the output and applies activation function
        let outputs = Matrix.multiply(this.weights_ho, hidden)
        outputs.add(this.bias_o)
        outputs.map(sigmoid)

        /**
         * Uses target matrix and final output to calc error
         */
        // Target array converted to vector
        let targets = Matrix.fromArray(target_array)
        // Calc error - ERROR = TARGETS - OUTPUTS
        let output_errors = Matrix.subtract(targets, outputs)

        /**
         * Back propagation
         */
        // Find gradient of each output Point on sigmoid graph - by differentiating sigmoid function
        let gradients = Matrix.map(outputs, dsigmoid)
        // Edit gradient based on error
        gradients.multiply(output_errors)
        // Reduce change in gradient for less extreme changes 
        gradients.multiply(this.learning_rate)

        // Transpose matrix of hidden layer ouput so it can be multiplied by the gradients
        let hiddenT = Matrix.transpose(hidden)

        // Change in H -> O weights 
        let weights_ho_deltas = Matrix.multiply(gradients, hiddenT)
        this.weights_ho.add(weights_ho_deltas)
        // Add changes to  H -> O bias
        this.bias_o.add(gradients)


        
        // Transpose H -> O weights. Find error of hidden layer
        let weights_hoT = Matrix.transpose(this.weights_ho)
        let hidden_errors = Matrix.multiply(weights_hoT, output_errors)

        // Repeat gradient calculations and weight changes but for I -> H
        let hidden_gradient = Matrix.map(hidden, dsigmoid)
        hidden_gradient.multiply(hidden_errors)
        hidden_gradient.multiply(this.learning_rate)

        let inputsT = Matrix.transpose(inputs)
        let weights_ih_deltas = Matrix.multiply(hidden_gradient, inputsT)
        this.weights_ih.add(weights_ih_deltas)
        this.bias_h.add(hidden_gradient)
    }
}