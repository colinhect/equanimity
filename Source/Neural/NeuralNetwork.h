#pragma once

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class NeuralNetworkBuildError :
    public std::runtime_error
{
public:
    explicit NeuralNetworkBuildError(const std::string& message) :
        std::runtime_error(std::string("Failed to build neural network: ") + message)
    {
    }
};

class Layer
{
public:
    Layer()
    {
    }

    Layer(unsigned size) :
        _size(size)
    {
    }

    Layer(Layer& previousLayer, unsigned size) :
        _previousLayer(&previousLayer),
        _size(size)
    {
    }

    virtual ~Layer() { }

    unsigned GetSize() const
    {
        return _size;
    }

private:
    Layer* _previousLayer{ nullptr };
    unsigned _size{ 0 };
};

class Builder
{
public:
    virtual std::unique_ptr<Layer> Build()
    {
        throw NeuralNetworkBuildError("Builder requires previous layer");
        return std::unique_ptr<Layer>();
    }

    virtual std::unique_ptr<Layer> Build(Layer& previousLayer)
    {
        (void)previousLayer;
        throw NeuralNetworkBuildError("Builder requires no previous layer");
        return std::unique_ptr<Layer>();
    }
};

class InputLayer :
    public Layer
{
    friend class InputBuilder;
public:

    InputLayer(unsigned size) :
        Layer(size),
        _values(size, 0.0f)
    {
    }

private:
    unsigned _size;
    std::vector<float> _values;
};

class InputBuilder :
    public Builder
{
public:
    InputBuilder& Size(unsigned size)
    {
        _size = size;
        return *this;
    }

    std::unique_ptr<Layer> Build() override
    {
        return std::make_unique<InputLayer>(_size);
    }

private:
    unsigned _size{ 0 };
};

class FullyConnectedLayer :
    public Layer
{
    friend class FullyConnectedBuilder;
public:

    FullyConnectedLayer(Layer& previousLayer, unsigned size) :
        Layer(previousLayer, size),
        _weights(previousLayer.GetSize() * GetSize(), 0.0f)
    {
    }

private:
    std::vector<float> _weights;
};

class FullyConnectedBuilder :
    public Builder
{
public:
    FullyConnectedBuilder& Size(unsigned size)
    {
        _size = size;
        return *this;
    }

    std::unique_ptr<Layer> Build(Layer& previousLayer) override
    {
        unsigned size = _size;
        if (size == 0)
        {
            size = previousLayer.GetSize();
        }

        return std::make_unique<FullyConnectedLayer>(previousLayer, size);
    }

private:
    unsigned _size{ 0 };
};

class ActivationLayer :
    public Layer
{
public:
    ActivationLayer(Layer& previousLayer, unsigned size) :
        Layer(previousLayer, size)
    {
    }
};

class SigmoidActivationLayer :
    public ActivationLayer
{
public:
    SigmoidActivationLayer(Layer& previousLayer, unsigned size) :
        ActivationLayer(previousLayer, size),
        _biases(size, 0.0f)
    {
    }

private:
    std::vector<float> _biases;
};

class ActivationBuilder :
    public Builder
{
public:
    ActivationBuilder& Size(unsigned size)
    {
        _size = size;
        return *this;
    }

    ActivationBuilder& Sigmoid()
    {
        _sigmoid = true;
        return *this;
    }

    std::unique_ptr<Layer> Build(Layer& previousLayer) override
    {
        unsigned size = _size;
        if (size == 0)
        {
            size = previousLayer.GetSize();
        }

        if (_sigmoid)
        {
            return std::make_unique<SigmoidActivationLayer>(previousLayer, size);
        }
        else
        {
            throw NeuralNetworkBuildError("No activation layer type specified");
        }
    }

private:
    unsigned _size{ 0 };
    bool _sigmoid{ false };
};

class NeuralNetwork
{
    friend class NeuralNetworkBuilder;
public:

private:
    NeuralNetwork(std::vector<std::unique_ptr<Layer>>&& layers) :
        _layers(layers)
    {
    }

    std::vector<std::unique_ptr<Layer>> _layers;
};

class NeuralNetworkBuilder
{
public:
    NeuralNetworkBuilder& Layer(InputBuilder& builder)
    {
        _layers.emplace_back(builder.Build());
        return *this;
    }

    NeuralNetworkBuilder& Layer(Builder& builder)
    {
        if (_layers.empty())
        {
            _layers.emplace_back(builder.Build());
        }
        else
        {
            _layers.emplace_back(builder.Build(*_layers.back()));
        }

        return *this;
    }

    NeuralNetwork Build()
    {
        return NeuralNetwork(std::move(_layers));
    }

private:
    std::vector<std::unique_ptr<::Layer>> _layers;
};
