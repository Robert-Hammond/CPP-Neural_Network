/**
 * welcome to the MEAT and POTATOES, my friends.
 */
#include "neuralnetwork.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
/**
 * Argument Constructor for new neural networks.
 */
NeuralNetwork::NeuralNetwork(std::vector<unsigned int> the_num_layer_nodes)
{
    // check parameters and assign variables
    if (!checkParameters(the_num_layer_nodes))
        std::cerr << "Must have at least 2 layers and no layers can have 0 nodes." << std::endl;
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
        for (int node = 0; node < num_layer_nodes[layer]; ++node)
        {
            weights[layer][node] = new float[num_layer_nodes[layer - 1]];
            // randomize weights
            for (int prev_node = 0; prev_node < num_layer_nodes[layer - 1]; ++prev_node)
                weights[layer][node][prev_node] = getRandomWeight();
            // randomize biases
            biases[layer][node] = getRandomBias();
        }
    }
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

/**
 * Saves the valuable information (them finely crafted weights and biases)
 * to a text file.
 */
void NeuralNetwork::save() const
{
    std::ofstream out(file_name.c_str());

    // write dimensions into the file
    out << "structure: ";
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
            out << std::endl << biases[layer][node] << std::endl;
            // now the weights
            for (int preceding_node = 0; preceding_node < num_layer_nodes[layer - 1]; ++preceding_node)
                out << weights[layer][node][preceding_node] << " ";
        }
    }
    out.close();
}

/**
 * Run an input through the network and return the output.
 * If is_training is not specified, it is assumed this input should not
 * backpropogate.
 */