/**
 * Header for the NeuralNet
 */
#include <string>
#include <vector>
class NeuralNetwork
{
public:
    // Argument constructor for a new net
    NeuralNetwork(std::vector<unsigned int> the_num_layer_nodes);
    // Argument constructor for an existing net
    NeuralNetwork(const std::string &the_file_name);

    // save
    void save() const;

    // public member functions
    float *run(float *input, bool is_training = false);

private:
    // Important to keep this in a member variable, this file will store
    // all of the weights and biases when the Net is saved.
    std::string file_name;
    bool saved;
    unsigned int num_layers;
    std::vector<unsigned int> num_layer_nodes;

    // REPRESENTATION
    float **activations;
    float ***weights;
    float **biases;

    // CONSTANTS
    const float MIN_WEIGHT = -10, MAX_WEIGHT = 10,
                MIN_BIAS = -10, MAX_BIAS = 10;

    // PRIVATE METHODS
    bool checkParameters(std::vector<unsigned int> the_num_layer_nodes) const;
    float getRandomWeight() const;
    float getRandomBias() const;
};