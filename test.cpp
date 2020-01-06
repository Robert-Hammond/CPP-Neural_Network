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
    test_constructors();
    test_run();
    test_learning();
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
    float test_input1[1] = {0.5};
    float *output = n1.run(test_input1);
    assert(fabs(output[0] - 0.924141819979) < 0.001);
    // trying out a network with a hidden layer and 2 nodes
    NeuralNetwork n2("test_run_2.txt");
    float test_input2[1] = {0.7};
    output = n2.run(test_input2);
    assert(fabs(output[0] - 0.040269462466) < 0.001);
}

int classification(int size, float *output)
{
    int max = 0;
    for (int i = 0; i < size; ++i)
    {
        if (output[i] > output[max])
            max = i;
    }
    return max;
}

void test_learning(const TRAINING_DATA &data, const std::vector<unsigned int> &structure, float lr = 0.03)
{
    NeuralNetwork net(structure, lr);
    // divide up the data into training and testing portions
    unsigned int training_size = (int)(0.8 * data.size());
    float tracker = 0;
    // give the network a chance to learn
    for (unsigned int i = 0; i < training_size; ++i)
    {
        float *input = data[i].first;
        std::cout << input[0] << "   ";
        float *expected_output = data[i].second;
        net.run(input, expected_output);
        if (i % 20 == 0)
        {
            std::cout << "cost tracker = " << tracker << std::endl;
            tracker = 0;
        }
        tracker += net.cost(expected_output);
    }

    // use the testing data to test
    int num_correct = 0, output_size = structure[structure.size() - 1];
    for (unsigned int i = training_size; i < data.size(); ++i)
    {
        float *input = data[i].first;
        float *output = net.run(input);
        float *expected_output = data[i].second;
        std::cout << "input: " << input[0] << " , " << input[1] << std::endl;
        std::cout << output[0] << " , " << output[1] << " vs " << expected_output[0] << " , " << expected_output[1] << "\n";
        //std::cout << classification(output_size, output) << std::endl;
        if (classification(output_size, output) == classification(output_size, expected_output))
            ++num_correct;
    }
    // print the results
    std::cout << (float(num_correct) * 100.0 / float(data.size() - training_size)) << "% accuracy\n\n";
}

void test_learning()
{
    // generate data following a simple pattern:
    // if the number is positive, classify the first, otherwise the second
    std::vector<std::pair<float *, float *>> data;
    srand(time(nullptr));
    for (int i = 0; i < 100; ++i)
    {
        float input[1] = {(float(rand()) / INT32_MAX) * 2 - 1};
        float output[2] = {float(input[0] > 0.0 ? 1.0 : 0.0), float(input[0] > 0.0 ? 0.0 : 1.0)};
        data.push_back(std::make_pair(input, output));
    }
    std::vector<unsigned int> structure = {1, 2, 2, 2};
    test_learning(data, structure, 0.01);

    // if the two numbers have the same sign, classify the second, otherwise the first
    data.clear();
    for (int i = 0; i < 100; ++i)
    {
        //srand(time(nullptr));
        float r1 = ((float)rand() / INT32_MAX) * 2 - 1;
        //srand(time(nullptr));
        float r2 = ((float)rand() / INT32_MAX) * 2 - 1;
        float input[2] = {r1, r2};
        float output[2] = {1.0, 0.0};
        if (r1 * r2 > 0)
            output[0] = 0.0;
        output[1] = 1 - output[0];
        data.push_back(std::make_pair(input, output));
    }
    structure = {2, 4, 4, 4, 2};
    test_learning(data, structure, 0.1);

    // if the two numbers are within ~0.8 of the origin
    data.clear();
    for (int i = 0; i < 100; ++i)
    {
        //srand(time(nullptr));
        float r1 = ((float)rand() / INT32_MAX) * 2 - 1;
        //srand(time(nullptr));
        float r2 = ((float)rand() / INT32_MAX) * 2 - 1;
        float input[2] = {r1, r2};
        float output[2] = {1.0, 0.0};
        if (pow(r1, 2) + pow(r2, 2) > 0.6366)
            output[0] = 0.0;
        output[1] = 1 - output[0];
        data.push_back(std::make_pair(input, output));
    }
    structure = {2, 4, 4, 2};
    test_learning(data, structure, 0.1);
}