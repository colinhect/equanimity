#pragma once

#include <vector>

#include "ComputationalGraph.h"

class MultilayerPerceptron :
    public ComputationalGraph
{
public:
    MultilayerPerceptron(const std::vector<unsigned>& layerCounts);
};
