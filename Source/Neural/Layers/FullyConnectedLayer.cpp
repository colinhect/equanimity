///////////////////////////////////////////////////////////////////////////////
// This source file is part of Equanimity.
//
// Copyright (c) 2017 Colin Hill
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
///////////////////////////////////////////////////////////////////////////////
#include "FullyConnectedLayer.h"

using namespace equanimity;

FullyConnectedLayer::Builder& FullyConnectedLayer::Builder::Size(unsigned size)
{
    _size = size;
    return *this;
}

std::unique_ptr<NeuralNetworkLayer> FullyConnectedLayer::Builder::Build(NeuralNetworkLayer& previousLayer)
{
    unsigned size = _size;
    if (size == 0)
    {
        size = previousLayer.GetSize();
    }

    return std::make_unique<FullyConnectedLayer>(previousLayer, size);
}

FullyConnectedLayer::FullyConnectedLayer(NeuralNetworkLayer& previousLayer, unsigned size) :
    NeuralNetworkLayer(previousLayer, size),
    _weights(previousLayer.GetSize() * GetSize(), 0.0f)
{
}
