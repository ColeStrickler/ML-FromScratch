#include "neuralnetwork.h"
#include <stdio.h>

static int getGuess(const LayerActivation &output_layer)
{
    int guess = 0;
    float highest = 0.00f;
    for (int i = 0; i < output_layer.size; i++)
    {
        auto activation = output_layer.activations[i];
        if (activation > highest)
        {
            highest = activation;
            guess = i;
        }
    }
    return guess;
}

DenseLayer::DenseLayer(int size, int input_size) : m_LayerSize(size), m_InputSize(input_size)
{
    m_Weights = new float[size * (input_size + 1)]; // each neuron needs a weight for each input + bias
    m_Activation = new float[size];                 // an output from each neuron
    printf("DenseLayer Size (%d, %d)\n", input_size, size);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < input_size + 1; j++)
        {
            m_Weights[i * (m_InputSize + 1) + j] = RandomFloatGenerator::get(); // randomize weights
                                                                                // printf("DenseLayer Size (%d, %d)\n", i, j);
        }
}

DenseLayer::~DenseLayer()
{
}

float *DenseLayer::FeedForward(float *input_activations, int size)
{
    assert(size == m_InputSize);

    // We will later allow switching method of inference

    // Basic Vector/Matrix Mul
    for (int neuron = 0; neuron < m_LayerSize; neuron++)
    {
        float dot_product = 0.0f;
        for (int weight = 0; weight < m_InputSize; weight++)
        {
            dot_product += input_activations[weight] * m_Weights[neuron * (m_InputSize + 1) + weight];
        }
        m_Activation[neuron] = sigmoid(dot_product + m_Weights[neuron * (m_InputSize + 1) + m_InputSize]); // dot product + bias
    }
    return m_Activation;
}

void DenseLayer::SetWeights(float *data, int size)
{
    assert(size == m_LayerSize * m_InputSize);
    for (int i = 0; i < m_InputSize; i++)
    {
        for (int neuron = 0; neuron < m_LayerSize; neuron++)
        {
            m_Weights[neuron * (m_InputSize + 1) + i] = data[i * m_LayerSize + neuron];
        }
    }
}

void DenseLayer::SetBiases(float *data, int size)
{
    assert(size == m_LayerSize);
    for (int i = 0; i < m_LayerSize; i++)
    {
        m_Weights[i * (m_InputSize + 1) + m_InputSize] = data[i];
    }
}

std::vector<int> DenseLayer::GetDimensions()
{
    return {m_LayerSize};
}

LayerActivation DenseLayer::GetLayerActivation()
{
    return {m_LayerSize, m_Activation};
}

NeuralNetwork::NeuralNetwork(std::vector<std::unique_ptr<NetworkLayer>> &layers) : m_Layers(std::move(layers))
{
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::SGD(std::vector<std::vector<float>> &training_data, int epochs, int mini_batch_size, int eta,
                        const std::vector<std::vector<float>> &test_data)
{

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        printf("Epoch %d/%d\n", epoch, epochs);
        std::random_device rd;
        std::mt19937 gen(rd()); // Random engine
        std::shuffle(training_data.begin(), training_data.end(), gen);

        for (int j = 0; j < training_data.size(); j += mini_batch_size)
        {
            std::vector<std::vector<float>> mini_batch(training_data.begin() + j, training_data.begin() + mini_batch_size);

            UpdateMiniBatch(mini_batch, eta);
        }

        Evaluate(test_data);
    }
}

int NeuralNetwork::Inference(std::vector<float> &input_instance)
{
    float *activation = input_instance.data();
    int activation_size = input_instance.size();
    auto output_layer = m_Layers[m_Layers.size()-1].get();
    for (auto &layer : m_Layers)
    {
        layer->FeedForward(activation, activation_size);
        auto layer_activation = layer->GetLayerActivation();
        activation = layer_activation.activations;
        activation_size = layer_activation.size;
    }

    return getGuess(output_layer->GetLayerActivation());
}

std::vector<float> NeuralNetwork::Evaluate(const std::vector<std::vector<float>> &test_data)
{
    return std::vector<float>();
}

void NeuralNetwork::Backpropagation(std::vector<float> &input_instance)
{
}

std::vector<LayerActivation> NeuralNetwork::FeedforwardTrain(std::vector<float> &input_instance)
{

    std::vector<LayerActivation> activations_all;
    float *activation = input_instance.data();
    int activation_size = input_instance.size();

    for (auto &layer : m_Layers)
    {
        layer->FeedForward(activation, activation_size);
        auto layer_activation = layer->GetLayerActivation();
        activation = layer_activation.activations;
        activation_size = layer_activation.size;
        activations_all.push_back(layer_activation);
    }

    return activations_all;
}

void NeuralNetwork::UpdateMiniBatch(const std::vector<std::vector<float>> &mini_batch, int eta)
{
}
