#pragma once

#include "Neural/NeuralNetwork.h"

int main()
{
    NeuralNetwork network = NeuralNetworkBuilder()
        .Layer(InputBuilder().Size(28 * 28))
        .Layer(FullyConnectedBuilder())
        .Layer(ActivationBuilder().Sigmoid())
        .Build();

    return 0;
}
