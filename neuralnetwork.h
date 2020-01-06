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
    // Destructor
    ~NeuralNetwork();
    // Delete file
    void delete_file();

    // save
    void save(const std::string &save_file);
    void save() const;

    // public member functions
    float *run(float *input, float *expected_output = nullptr);
    float cost(float *expected_output);

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

    // REPRESENTATION

    // The activation of node n in layer l is
    // activations[l][n]
    float **activations;
    // The weight from node m in layer l-1 to node n in layer l is
    // weights[l][n][m]
    float ***weights;
    // The bias of node n in layer l is
    // biases[l][n]
    float **biases;

    // CONSTANTS
    const float MIN_WEIGHT = -1, MAX_WEIGHT = 1,
                MIN_BIAS = -1, MAX_BIAS = 1;
    float learning_rate = 0.03;

    // PRIVATE HELPER METHODS
    float activation_function(float x) const;
    float activation_function_derivative(float x) const;
    float activation_function_inverse(float x) const;
    bool checkParameters(std::vector<unsigned int> the_num_layer_nodes) const;
    float getRandomWeight() const;
    float getRandomBias() const;
    void backpropogate(const float *expected_output);
};