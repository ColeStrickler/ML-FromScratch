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


std::vector<float> Hadamard(const std::vector<float>& a, const std::vector<float>& b)
{
    assert(a.size() == b.size());
    std::vector<float> ret;

    for (int i = 0; i < a.size(); i++)
        ret.push_back(a[i]*b[i]);
    return ret;
}

DenseLayer::DenseLayer(int size, int input_size) : m_LayerSize(size), m_InputSize(input_size)
{
    m_Weights = std::vector<float>(size * (input_size + 1)); // each neuron needs a weight for each input + bias
    m_Activation = std::vector<float>(size);                 // an output from each neuron
    z = std::vector<float>(size);
    printf("DenseLayer Size (%d, %d)\n", input_size, size);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < input_size + 1; j++)
        {
            m_Weights[i * (m_InputSize + 1) + j] = RandomFloatGenerator::get(); // randomize weights
                                                                                // printf("DenseLayer Size (%d, %d)\n", i, j);
        }
}

float DenseLayer::GetWeight(int src_neuron, int dst_neuron)
{
    assert(dst_neuron < m_LayerSize);
    assert(src_neuron < m_InputSize);
    return m_Weights[dst_neuron*(input_size+1) + src_neuron];
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
        z[neuron] = dot_product + m_Weights[neuron * (m_InputSize + 1) + m_InputSize];
        m_Activation[neuron] = sigmoid(z[neuron]); // dot product + bias
    }
    return m_Activation.data();
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
    return {m_InputSize, m_LayerSize};
}


LayerActivation DenseLayer::GetLayerActivation()
{
    return {m_LayerSize, m_Activation.data(), z.data()};
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

void NeuralNetwork::Backpropagation(std::vector<float> &input_instance, int label)
{
    BackPropResult ret;
    auto nabla_w = AllocGradientStorage();
    auto nabla_b = AllocBiasGradientStorage();

    /*
        We need to rethink this. Want to access like matrix[L][k][j]
    */


    auto activations = FeedforwardTrain(input_instance);
    auto output_layer_activation = activations[activations.size()-1];
    std::vector<float> delta_L = Hadamard(BaseCostDerivative(output_layer_activation, label),\
     applyToEachManual(output_layer_activation.z, output_layer_activation.size, sigmoid_prime));

    int L = m_Layers.size()-1;



    nabla_b[L] = delta_L;

    std::vector<float> last;
    for (int j = 0; j < delta_L.size(); j++) // activation of neuron_L-1_k to neuron_L_j
    {
        auto activation = activations[L-1];
        float tmp = 0.0f; // I think this is right
        for (int k = 0; k < activation.size; k++)
        {
            nabla_w[L][j][k] = delta_L[j]*activation.activations[k];
        }   
    }
    L--;


    while (L >= 0)
    {
        float* z = activations[L].z;
        int z_size = activations[L].size;
        auto sigmoid_prime_z = applyToEachManual(z, z_size, sigmoid_prime);
        
        auto prev_delta_L = delta_L;
        delta_L = std::vector<float>(z_size, 0.0f); // delta value for each neuron in layer
        
        for (int j = 0; j < delta_L.size(); j++)
        {
            for (int k = 0; k < prev_delta_L.size(); k++)
            {
                /*
                    delta_L_j = SUM[(w_L+1_k_j)(sigma_prime(z_L_j))(delta_L+1_k)
                */
                delta_L[j] += GetWeight(L+1, k, j)*sigma_prime(z[j])*prev_delta_L[k]; // need to get w_L+1_k_j
            }
        }
        nabla_b = delta_L;
        for (int j = 0; j < delta_L.size(); j++) // activation of neuron_L-1_k to neuron_L_j
        {
            auto activation = activations[L-1];
            float tmp = 0.0f; // I think this is right
            for (int k = 0; k < activation.size; k++)
            {
                nabla_w[L][j][k] = delta_L[j]*activation.activations[k];
            }   
        }
        L--;
    }
    
    ret.gradient = std::move(nabla_w);
    ret.bias_gradient = std::move(nabla_b);

    return ret;
}

std::vector<float> NeuralNetwork::BaseCostDerivative(LayerActivation output_layer_activation, int label)
{
    std::vector<float> ret;
    /*
        If our cost function is (1/2)(y(x) - a_j)^2, we compute dC/a_j
        
        dC/a_j = -(y(x) - a_j) = (a_j - y(x))
    */

    for (int i = 0; i < output_layer_activation.size; i++)
    {
        if (i == label)
        {
            ret.push_back(output_layer_activation.activations[i] - 1.0f);
        }
        else
        {
            ret.push_back(output_layer_activation.activations[i] - 0.0f);
        }
    }

    return ret;
}


// mat[layer][neuron]
std::vector<std::vector<float>> NeuralNetwork::AllocBiasGradientStorage()
{
    std::vector<std::vector<float>> ret;

    for (auto& layer: m_Layers)
    {
        auto dim = layer->GetDimensions();
        auto input_size = dim[0];
        auto num_neurons = dim[1];
        std::vector<float>layer_gradient = std::vector<float>(num_neurons);
        ret.push_back(layer_gradient);
    }


    return ret;
}

float NeuralNetwork::GetWeight(int layer, int dst_neuron, int src_neuron)
{
    auto layer = m_Layers[layer];

    return layer.GetWeight(dst_neuron, src_neuron);
}

// mat[layer][neuron_j][weight_k]
std::vector<std::vector<std::vector<float>>> NeuralNetwork::AllocGradientStorage()
{
    std::vector<std::vector<std::vector<float>>> ret;

    for (auto& layer: m_Layers)
    {
        auto dim = layer->GetDimensions();
        auto input_size = dim[0];
        auto num_neurons = dim[1];
        std::vector<std::vector<float>>layer_gradient(input_size, std::vector<float>(num_neurons));
        ret.push_back(layer_gradient);
    }


    return ret;
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

void NeuralNetwork::UpdateMiniBatch(const std::vector<std::pair<int, std::vector<float>>> &mini_batch, int eta)
{

    auto nabla_w = AllocGradientStorage();
    auto nabla_b = AllocBiasGradientStorage();

    for (auto& instance: mini_batch)
    {

        // we need there to be labels in the instance --> fix
        BackPropResult res = Backpropagation(instance.second, instance.first);


        // I think all we need to do now is sum the nablas, and then update the weights. Also add labels
    }
}
