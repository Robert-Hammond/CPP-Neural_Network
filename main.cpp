/**
 * Yes, this is main.
 * There has to be at least 1,000,000,000 files on the internet called "main.cpp".
 * This is one of them.
 */

using namespace std;
#include <string>
#include <iostream>
#include "neuralnetwork.h"

void test_new_network();

int main()
{
    test_new_network();
}

void test_new_network()
{
    // initialize a new net
    vector<unsigned int> structure;
    structure.push_back(3);
    structure.push_back(4);
    structure.push_back(2);
    NeuralNetwork net(structure);
    net.save();
}