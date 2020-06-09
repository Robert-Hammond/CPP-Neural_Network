/**
 * Testing file
 */

#include <string>
#include <iostream>
#include "neuralnetwork.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>

typedef std::vector<std::pair<float *, float *>> TRAINING_DATA;

void test_constructors();
void test_run();
void test_learning();

int main()
{
    std::cout << "================= BEGINNING TESTS =================" << std::endl;
    test_constructors();
    test_run();
    test_learning();
    std::cout << "================= TESTS FINISHED ==================" << std::endl;
}

void test_constructors()
{
    // initialize a new net
    std::vector<unsigned int> structure;
    structure.push_back(3);
    structure.push_back(4);
    structure.push_back(2);
    NeuralNetwork net1(structure);
    net1.save();

    // load a saved net
    NeuralNetwork net2(net1.getFileName());
    net1.delete_file();
}

/**
 * tests the run() operation without backpropogation.
 * All the expected outputs of these test networks were worked out
 * on paper beforehand.
 */
void test_run()
{
    // very straightforward to start, no hidden layers
    NeuralNetwork n1("test_run_1.txt");
    std::vector<float> test_input1 = {0.5};
    std::vector<float> output = n1.run(test_input1);
    // for sigmoid:
    // assert(fabs(output[0] - 0.924141819979) < 0.001);

    // trying out a network with a hidden layer and 2 nodes
    NeuralNetwork n2("test_run_2.txt");
    std::vector<float> test_input2 = {0.7};
    output = n2.run(test_input2);
    // for sigmoid:
    // assert(fabs(output[0] - 0.040269462466) < 0.001);
}

int classification(int size, std::vector<float> output)
{
    int max = 0;
    for (int i = 0; i < size; ++i)
    {
        if (output[i] > output[max])
            max = i;
    }
    return max;
}

void test_learning(unsigned int data_size, std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> outputs, const std::vector<unsigned int> &structure, float lr = 0.03)
{
    NeuralNetwork net(structure, lr);
    // divide up the data into training and testing portions
    unsigned int training_size = (int)(0.8 * data_size);
    float tracker = 0, samples = 0;
    // give the network a chance to learn
    for (unsigned int i = 0; i < training_size; ++i)
    {
        std::vector<float> input = inputs[i];
        std::vector<float> expected_output = outputs[i];
        net.run(input, expected_output);
        if (i % (training_size / 10) == 0 && i > 0)
        {
            std::cout << "avg cost = " << (tracker / samples) << std::endl;
            tracker = 0;
        }
        tracker += net.cost(expected_output);
        samples++;
    }

    // use the testing data to test
    int num_correct = 0, output_size = structure[structure.size() - 1];
    for (unsigned int i = training_size; i < data_size; ++i)
    {
        std::vector<float> input = inputs[i];
        std::vector<float> output = net.run(input);
        std::vector<float> expected_output = outputs[i];
        if (classification(output_size, output) == classification(output_size, expected_output))
            ++num_correct;
    }
    // print the results
    std::cout << (float(num_correct) * 100.0 / float(data_size - training_size)) << "% accuracy\n\n";
}

/**
 * generate data following various simple patterns, then test how well a network can
 * learn the pattern.
 */
void test_learning()
{
    unsigned int data_size = 300;
    srand(time(nullptr));
    // if the number is positive, classify the first, otherwise the second
    std::vector<std::vector<float>> input(data_size);
    std::vector<std::vector<float>> output(data_size);
    for (unsigned int i = 0; i < data_size; ++i)
    {
        input[i] = {(float(rand()) / INT32_MAX) * 2 - 1};
        output[i] = {float(input[i][0] > 0.0 ? 1.0 : -1.0), float(input[i][0] > 0.0 ? -1.0 : 1.0)};
    }
    std::vector<unsigned int> structure = {1, 2, 2};
    test_learning(data_size, input, output, structure, 0.03);

    data_size = 1000000;
    input.resize(data_size);
    output.resize(data_size);
    // if the two numbers have the same sign, classify the second, otherwise the first
    for (unsigned int i = 0; i < data_size; ++i)
    {
        float r1 = ((float)rand() / INT32_MAX) * 2 - 1;
        float r2 = ((float)rand() / INT32_MAX) * 2 - 1;
        input[i] = {r1, r2};
        output[i] = {1.0, -1.0};
        if (r1 * r2 > 0)
            output[i][0] = -1.0;
        output[i][1] = -output[i][0];
    }
    structure = {2, 2, 2};
    test_learning(data_size, input, output, structure, 0.1);

    data_size = 1000000;
    input.resize(data_size);
    output.resize(data_size);
    // if the two numbers are within ~0.8 of the origin
    for (unsigned int i = 0; i < data_size; ++i)
    {
        float r1 = ((float)rand() / INT32_MAX) * 2 - 1;
        float r2 = ((float)rand() / INT32_MAX) * 2 - 1;
        input[i] = {r1, r2};
        output[i] = {1.0, -1.0};
        if (pow(r1, 2) + pow(r2, 2) > 0.6366)
            output[i][0] = -1.0;
        output[i][1] = -output[i][0];
    }
    structure = {2, 4, 2, 2};
    test_learning(data_size, input, output, structure, 0.1);
}