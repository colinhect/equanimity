#pragma once

#include <CL/cl.hpp>

#include "Neural/NeuralNetwork.h"
#include "Neural/NeuralNetworkBuilder.h"
#include "Neural/Layers/ActivationLayer.h"
#include "Neural/Layers/FullyConnectedLayer.h"
#include "Neural/Layers/InputLayer.h"
#include "Neural/Layers/OutputLayer.h"

int main()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    

    return 0;
}

void NeuralNetwork()
{
    using namespace equanimity;
    NeuralNetworkBuilder networkBuilder;
    networkBuilder.AddLayer<InputLayer>().Size(28 * 28);
    networkBuilder.AddLayer<FullyConnectedLayer>();
    networkBuilder.AddLayer<ActivationLayer>().Sigmoid();
    networkBuilder.AddLayer<FullyConnectedLayer>();
    networkBuilder.AddLayer<OutputLayer>().Size(10);

    NeuralNetwork network = networkBuilder.Build();
}

