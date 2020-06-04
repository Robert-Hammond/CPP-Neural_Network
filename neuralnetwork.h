/**
 * Header for the NeuralNet
 */
#include <string>
#include <vector>
class NeuralNetwork
{
public:
    // Argument constructor for a new net
    NeuralNetwork(std::vector<unsigned int> the_num_layer_nodes, float lr = 0.03);
    // Argument constructor for an existing net
    NeuralNetwork(const std::string &the_file_name);
    // Copy constructor
    NeuralNetwork(const NeuralNetwork &other);
    // Delete file
    void delete_file();

    // save
    void save(const std::string &save_file);
    void save() const;

    // public member functions
    // includes backpropogation
    std::vector<float> run(std::vector<float> input, std::vector<float> expected_output);
    // does not backpropogate
    std::vector<float> run(std::vector<float> input);
    float cost(std::vector<float> expected_output);

    // accessors
    const std::string &getFileName() const { return file_name; }
    const std::vector<unsigned int> &getStructure() const { return num_layer_nodes; }

private:
    // Important to keep this in a member variable, this file will store
    // all of the weights and biases when the Net is saved.
    std::string file_name;
    bool saved;
    unsigned int num_layers;
    std::vector<unsigned int> num_layer_nodes;
    std::string activation_function_name;

    // REPRESENTATION

    // The activation of node n in layer l is
    // activations[l][n]
    std::vector<std::vector<float>> activations;
    // The weight from node m in layer l-1 to node n in layer l is
    // weights[l][n][m]
    std::vector<std::vector<std::vector<float>>> weights;
    // The bias of node n in layer l is
    // biases[l][n]
    std::vector<std::vector<float>> biases;

    // CONSTANTS
    const float MIN_WEIGHT = -0.5, MAX_WEIGHT = 0.5,
                MIN_BIAS = -0.5, MAX_BIAS = 0.5;
    float learning_rate = 0.03;

    // PRIVATE HELPER METHODS
    float activation_function(float x) const;
    float activation_function_derivative(float x) const;
    float activation_function_inverse(float x) const;
    bool checkParameters(std::vector<unsigned int> the_num_layer_nodes) const;
    float getRandomWeight() const;
    float getRandomBias() const;
    void backpropogate(const std::vector<float> expected_output);
};