#pragma once

#include <cassert>
#include <memory>
#include <vector>

class Layer
{
public:
    class Builder
    {
    public:
        virtual std::unique_ptr<Layer> Build(Layer* previousLayer) = 0;
    };

    Layer()
    {
    }

    Layer(Layer& previousLayer) :
        _previousLayer(&previousLayer)
    {
    }

    virtual ~Layer() { }

    virtual unsigned GetSize() const
    {
        if (_previousLayer)
        {
            return _previousLayer->GetSize();
        }
        else
        {
            return 0;
        }
    }

private:
    Layer* _previousLayer{ nullptr };
};

class InputLayer :
    public Layer
{
public:
    class Builder :
        public Layer::Builder
    {
    public:
        Builder& Size(unsigned size)
        {
            _size = size;
            return *this;
        }

        std::unique_ptr<Layer> Build(Layer* previousLayer) override
        {
            assert(!previousLayer);
            return std::make_unique<InputLayer>(_size);
        }

    private:
        unsigned _size{ 0 };
    };

    InputLayer(unsigned size) :
        _size(size),
        _values(size, 0.0f)
    {
    }

    virtual unsigned GetSize() const override
    {
        return _size;
    }

private:
    unsigned _size;
    std::vector<float> _values;
};

class FullyConnectedLayer :
    public Layer
{
public:
    class Builder :
        public Layer::Builder
    {
    public:
        Builder& Size(unsigned size)
        {
            _size = size;
            return *this;
        }

        std::unique_ptr<Layer> Build(Layer* previousLayer) override
        {
            assert(previousLayer);

            unsigned size = _size;
            if (size == 0)
            {
                size = previousLayer->GetSize();
            }

            return std::make_unique<FullyConnectedLayer>(*previousLayer, size);
        }

    private:
        unsigned _size{ 0 };
    };

    FullyConnectedLayer(Layer& previousLayer, unsigned size) :
        _size(size),
        _weights(previousLayer.GetSize() * _size, 0.0f)
    {
    }

    virtual unsigned GetSize() const override
    {
        return _size;
    }

private:
    unsigned _size{ 0 };
    std::vector<float> _weights;
};

class ActivationLayer :
    public Layer
{
public:
    ActivationLayer()
    {
    }
};

class SigmoidActivationLayer :
    public ActivationLayer
{
public:
    SigmoidActivationLayer()
    {
    }

private:
    std::vector<float> _biases;
};

class MultilayerPerceptron
{
public:
    MultilayerPerceptron()
    {
    }

    void BuildLayer(Layer::Builder& builder)
    {
        Layer* previousLayer = nullptr;
        if (!_layers.empty())
        {
            previousLayer = _layers.back().get();
        }

        _layers.emplace_back(builder.Build(previousLayer));
    }

private:
    std::vector<std::unique_ptr<Layer>> _layers;
};
