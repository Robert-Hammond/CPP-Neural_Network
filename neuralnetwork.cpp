/**
 * welcome to the MEAT and POTATOES, my friends.
 */
#include "neuralnetwork.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>

/**
 * Argument Constructor for new neural networks.
 */
NeuralNetwork::NeuralNetwork(std::vector<unsigned int> the_num_layer_nodes, float lr)
{
    // check parameters and assign variables
    if (!checkParameters(the_num_layer_nodes))
    {
        std::cerr << "Must have at least 2 layers and no layers can have 0 nodes." << std::endl;
        exit(1);
    }
    num_layer_nodes = the_num_layer_nodes;
    num_layers = num_layer_nodes.size();
    learning_rate = lr;

    // make a new file to store this net with a random file name
    srand(time(nullptr));
    int rand_int = rand();
    file_name = "neural_network_" + std::to_string(rand_int) + ".txt";

    // initialize variables with random weights and biases
    activations.resize(num_layers);
    // for both of the weights and biases, we only need to accout for num_layers - 1 layers,
    // but for the sake of consistent notation with activations we will just ingore the 0th slot
    weights.resize(num_layers);
    biases.resize(num_layers);
    activations[0].resize(num_layer_nodes[0]);

    for (unsigned int layer = 1; layer < num_layers; ++layer)
    {
        activations[layer].resize(num_layer_nodes[layer]);
        biases[layer].resize(num_layer_nodes[layer]);
        weights[layer].resize(num_layer_nodes[layer]);
        for (unsigned int node = 0; node < num_layer_nodes[layer]; ++node)
        {
            weights[layer][node].resize(num_layer_nodes[layer - 1]);
            // randomize weights
            for (unsigned int prev_node = 0; prev_node < num_layer_nodes[layer - 1]; ++prev_node)
                weights[layer][node][prev_node] = getRandomWeight();
            // randomize biases
            biases[layer][node] = getRandomBias();
        }
    }
}

/**
 * File constructor for Neural Networks.
 */
NeuralNetwork::NeuralNetwork(const std::string &the_file_name)
{
    file_name = the_file_name;
    std::ifstream istr(file_name);
    if (!istr.good())
    {
        std::cerr << "Invalid file." << std::endl;
        exit(0);
    }

    std::string token;
    int n;
    istr >> token >> learning_rate; // token is "learning_rate:"
    istr >> token >> num_layers;    // token is "structure:"
    for (unsigned int i = 0; i < num_layers; ++i)
    {
        istr >> n;
        num_layer_nodes.push_back(n);
    }

    activations.resize(num_layers);
    weights.resize(num_layers);
    biases.resize(num_layers);
    activations[0].resize(num_layer_nodes[0]);

    // read all weights and biases
    for (unsigned int layer = 1; layer < num_layers; ++layer)
    {
        activations[layer].resize(num_layer_nodes[layer]);
        biases[layer].resize(num_layer_nodes[layer]);
        weights[layer].resize(num_layer_nodes[layer]);
        for (unsigned int node = 0; node < num_layer_nodes[layer]; ++node)
        {
            weights[layer][node].resize(num_layer_nodes[layer - 1]);
            // the biases first
            istr >> biases[layer][node];
            // now the weights
            for (unsigned int preceding_node = 0; preceding_node < num_layer_nodes[layer - 1]; ++preceding_node)
                istr >> weights[layer][node][preceding_node];
        }
    }
}

/**
 * The activation function is applied to each node after adding up all outputs
 * from all the connections of the previous layer.
 * Right now, I'm going to use a sigmoid because this is a learning exercise.
 * This might change in the future, and I might add a parameter that identifies
 * which activation function to use (i.e. Sigmoid, ReLU, tanh, you name it).
 * @return a float between 0 and 1 (0 for more negative inputs and 1 for more
 * positive inputs).
 * @param x any floating point number
 */
float NeuralNetwork::activation_function(float x) const
{
    return 1.0 / (1.0 + exp(-x));
}

float NeuralNetwork::activation_function_derivative(float x) const
{
    float activation = activation_function(x);
    return activation * (1 - activation);
}

float NeuralNetwork::activation_function_inverse(float x) const
{
    if (x <= 0)
        return -99999;
    if (x >= 1)
        return 99999;
    return -log((1.0 / x) - 1.0);
}

/**
 * Run an input through the network and return the output.
 */
std::vector<float> NeuralNetwork::run(std::vector<float> input)
{
    for (unsigned int i = 0; i < num_layer_nodes[0]; ++i)
        activations[0][i] = input[i];
    // layer is that which we are updating the values, starting at 1
    for (unsigned int layer = 1; layer < num_layers; ++layer)
    {
        for (unsigned int node = 0; node < num_layer_nodes[layer]; ++node)
        {
            float total = 0;
            for (unsigned int prev_node = 0; prev_node < num_layer_nodes[layer - 1]; ++prev_node)
            {
                total += weights[layer][node][prev_node] * activations[layer - 1][prev_node];
            }
            total = activation_function(total + biases[layer][node]);
            activations[layer][node] = total;
        }
    }
    return activations[num_layers - 1];
}

std::vector<float> NeuralNetwork::run(std::vector<float> input, std::vector<float> expected_output)
{
    std::vector<float> result = run(input);
    if (!expected_output.empty())
        backpropogate(expected_output);
    return result;
}

float NeuralNetwork::cost(std::vector<float> expected_output)
{
    float sum = 0;
    for (unsigned int node = 0; node < num_layer_nodes[num_layers - 1]; ++node)
        sum += pow(activations[num_layers - 1][node] - expected_output[node], 2);
    return sum;
}

/**
 * Must have at least 2 layers.
 * Also, no zeros in the node counts. That makes for a useless Neural Network.
 */
bool NeuralNetwork::checkParameters(std::vector<unsigned int> the_num_layer_nodes) const
{
    for (unsigned int n : the_num_layer_nodes)
        if (n == 0)
            return false;
    return the_num_layer_nodes.size() >= 2;
}

// random float in the range min to max
float random(float min, float max)
{
    return min + float(rand()) / (float(RAND_MAX / (max - min)));
}

/**
 * returns a random float between MIN_WEIGHT and MAX_WEIGHT.
 */
float NeuralNetwork::getRandomWeight() const
{
    return random(MIN_WEIGHT, MAX_WEIGHT);
}

/**
 * returns a random float between MIN_BIAS and MAX_BIAS
 */
float NeuralNetwork::getRandomBias() const
{
    return random(MIN_BIAS, MAX_BIAS);
}

void NeuralNetwork::save(const std::string &save_file)
{
    file_name = save_file;
    save();
}

/**
 * Saves the valuable information (them finely crafted weights and biases)
 * to a text file.
 */
void NeuralNetwork::save() const
{
    std::ofstream out(file_name.c_str());

    // write learning rate and dimensions into the file
    out << "learning_rate: " << learning_rate << std::endl
        << "structure: " << num_layers << std::endl;
    for (unsigned int i = 0; i < num_layers; ++i)
    {
        out << num_layer_nodes[i] << " ";
    }
    out << std::endl;

    // write all weights and biases
    for (unsigned int layer = 1; layer < num_layers; ++layer)
    {
        out << std::endl;
        for (unsigned int node = 0; node < num_layer_nodes[layer]; ++node)
        {
            // the biases first
            out << std::endl
                << biases[layer][node] << std::endl;
            // now the weights
            for (unsigned int preceding_node = 0; preceding_node < num_layer_nodes[layer - 1]; ++preceding_node)
                out << weights[layer][node][preceding_node] << " ";
        }
    }
    out.close();
}

void NeuralNetwork::delete_file()
{
    remove(file_name.c_str());
}

/**
 * Backpropogation algorithm was implemented with great help from
 * https://en.wikipedia.org/wiki/Backpropagation
 */
void NeuralNetwork::backpropogate(const std::vector<float> expected_output)
{
    std::vector<std::vector<float>> delta_values(num_layers);
    // starting from the output layer going backwards
    for (unsigned int layer = num_layers - 1; layer >= 1; --layer)
    {
        delta_values[layer].resize(num_layer_nodes[layer]);
        // node_j is of the current layer
        for (unsigned int node_j = 0; node_j < num_layer_nodes[layer]; ++node_j)
        {
            // calculate the net input from previous functions
            float net_j = activation_function_inverse(activations[layer][node_j]);

            // calculate the delta value of this node
            float delta_j = activation_function_derivative(net_j);
            if (layer == num_layers - 1)
            {
                delta_j *= 2 * (activations[layer][node_j] - expected_output[node_j]);
            }
            else
            {
                float sum = 0;
                // node_l is in the next layer
                for (unsigned int node_l = 0; node_l < num_layer_nodes[layer + 1]; ++node_l)
                {
                    sum += weights[layer + 1][node_l][node_j] * delta_values[layer + 1][node_l];
                }
                delta_j *= sum;
            }
            delta_values[layer][node_j] = delta_j;

            // node_i is of the previous layer
            for (unsigned int node_i = 0; node_i < num_layer_nodes[layer - 1]; ++node_i)
            {
                // calculate the derivative of cost with respect to this weight
                float dEdW = activations[layer - 1][node_i] * delta_j;
                // apply this derivative to modify the weight
                weights[layer][node_j][node_i] -= learning_rate * dEdW;
            }
            // the derivate of cost with respect to each bias is equal to the node's delta value
            biases[layer][node_j] -= learning_rate * delta_j;
        }
    }
}