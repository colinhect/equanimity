#pragma once

#include "Neural/MultilayerPerceptron.h"

int main()
{
    MultilayerPerceptron perceptron;
    perceptron.BuildLayer(InputLayer::Builder().Size(28 * 28));
    perceptron.BuildLayer(FullyConnectedLayer::Builder());

    return 0;
}

