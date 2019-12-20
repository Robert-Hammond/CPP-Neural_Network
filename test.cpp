/**
 * Testing file
 */

#include <string>
#include <iostream>
#include "neuralnetwork.h"
#include <cassert>
#include <cmath>

void test_constructors();
void test_run();

int main()
{
    test_constructors();
    test_run();
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