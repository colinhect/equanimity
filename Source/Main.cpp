#pragma once

#include "Neural/NeuralNetwork.h"
#include "Neural/NeuralNetworkBuilder.h"
#include "Neural/Layers/ActivationLayer.h"
#include "Neural/Layers/FullyConnectedLayer.h"
#include "Neural/Layers/InputLayer.h"

int main()
{
    using namespace equanimity;

    NeuralNetworkBuilder networkBuilder;
    networkBuilder.AddLayer<InputLayer>().Size(28 * 28);
    networkBuilder.AddLayer<FullyConnectedLayer>();
    networkBuilder.AddLayer<ActivationLayer>().Sigmoid();

    NeuralNetwork network = networkBuilder.Build();

    return 0;
}
