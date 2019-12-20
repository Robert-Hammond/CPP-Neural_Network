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
NeuralNetwork::NeuralNetwork(std::vector<unsigned int> the_num_layer_nodes)
{
    // check parameters and assign variables
    if (!checkParameters(the_num_layer_nodes))
    {
        std::cerr << "Must have at least 2 layers and no layers can have 0 nodes." << std::endl;
        exit(1);
    }
    num_layer_nodes = the_num_layer_nodes;
    num_layers = num_layer_nodes.size();

    // make a new file to store this net with a random file name
    srand(time(NULL));
    int rand_int = rand();
    file_name = "neural_network_" + std::to_string(rand_int) + ".txt";

    // initialize variables with random weights and biases
    activations = new float *[num_layers];
    // for both of the weights and biases, we only need to accout for num_layers - 1 layers,
    // but for the sake of consistent notation with activations we will just ingore the 0th slot
    weights = new float **[num_layers];
    biases = new float *[num_layers];
    activations[0] = new float[num_layer_nodes[0]];

    for (unsigned int layer = 1; layer < num_layers; ++layer)
    {
        activations[layer] = new float[num_layer_nodes[layer]];
        biases[layer] = new float[num_layer_nodes[layer]];
        weights[layer] = new float *[num_layer_nodes[layer]];
        for (unsigned int node = 0; node < num_layer_nodes[layer]; ++node)
        {
            weights[layer][node] = new float[num_layer_nodes[layer - 1]];
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

    istr >> token >> num_layers; // token is "structure:"
    for (unsigned int i = 0; i < num_layers; ++i)
    {
        istr >> n;
        num_layer_nodes.push_back(n);
    }

    activations = new float *[num_layers];
    weights = new float **[num_layers];
    biases = new float *[num_layers];
    activations[0] = new float[num_layer_nodes[0]];

    // read all weights and biases
    for (unsigned int layer = 1; layer < num_layers; ++layer)
    {
        activations[layer] = new float[num_layer_nodes[layer]];
        biases[layer] = new float[num_layer_nodes[layer]];
        weights[layer] = new float *[num_layer_nodes[layer]];
        for (unsigned int node = 0; node < num_layer_nodes[layer]; ++node)
        {
            weights[layer][node] = new float[num_layer_nodes[layer - 1]];
            // the biases first
            istr >> biases[layer][node];
            // now the weights
            for (unsigned int preceding_node = 0; preceding_node < num_layer_nodes[layer - 1]; ++preceding_node)
                istr >> weights[layer][node][preceding_node];
        }
    }
}

NeuralNetwork::~NeuralNetwork()
{
    delete[] activations[0];
    for (unsigned int layer = 1; layer < num_layers; ++layer)
    {
        delete[] activations[layer];
        delete[] biases[layer];
        for (unsigned int node = 0; node < num_layer_nodes[layer]; ++node)
            delete[] weights[layer][node];
        delete[] weights[layer];
    }
    delete[] activations;
    delete[] biases;
    delete[] weights;
}

/**
 * The activation function is applied to each node after adding up all outputs
 * from all the connections of the previous layer.
 * Right now, I'm going to use a sigmoid because this is a learning exercise.
 * This might change in the future, and I might add a parameter that identifies
 * which activation function to use (i.e. Sigmoid, ReLU, arctan, you name it).
 * @return a float between 0 and 1 (0 for more negative inputs and 1 for more
 * positive inputs).
 * @param x any floating point number
 */
float NeuralNetwork::activation_function(float x) const
{
    return 1.0 / (1.0 + exp(-x));
}

/**
 * Run an input through the network and return the output.
 * If is_training is not specified, it is assumed this input should not
 * backpropogate.
 */
float *NeuralNetwork::run(float *input, bool is_training)
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
    if (is_training)
        backpropogate(input);
    return activations[num_layers - 1];
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
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
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

    // write dimensions into the file
    out << "structure: " << num_layers << std::endl;
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
 * Still in development
 */
void NeuralNetwork::backpropogate(const float *input)
{
    ;
}