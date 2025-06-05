#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <vector>
#include <string> 
#include <random>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <memory>


enum LayerType
{
    DENSE,

};

template<typename T, typename Func>
std::vector<T> applyToEach(const std::vector<T>& vec, Func f) {
    std::vector<T> result;
    result.reserve(vec.size());
    std::transform(vec.begin(), vec.end(), std::back_inserter(result), f);
    return result;
}

template<typename T, typename Func>
std::vector<T> applyToEachManual(T* vec, int vecSize, Func f) {
    std::vector<T> result;
    result.reserve(vecSize);
    for (int i = 0; i < vecSize; i++)
        result[i] = f(vec[i]);
    return result;
}



inline float sigmoid(float val)
{
    return 1.0f/(1.0 + std::exp(-val));
}

inline float sigmoid_prime(float val)
{
    return sigmoid(val)*(1-sigmoid(val));
}

class RandomFloatGenerator {
public:
    // Generate a random float in [min, max)
    static float get(float min = 0.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(getEngine());
    }

private:
    // Static function that returns a reference to the shared engine
    static std::mt19937& getEngine() {
        static std::random_device rd;
        static std::mt19937 engine(rd());
        return engine;
    }
};

// refactor later to take this structure as input to feedforward
struct LayerActivation
{
    int size;
    float* activations;
    float* z;
};




class NetworkLayer
{
public:
    LayerType m_Type;
    virtual float* FeedForward(float* input_activations, int size) = 0; // pure virtual 
    virtual ~NetworkLayer() = default;  // Always add virtual destructor for base classes
    virtual LayerActivation GetLayerActivation() = 0; // pure virtual 
    virtual std::vector<int> GetDimensions() = 0;
    virtual float GetWeight(int src_neuron, int dst_neuron) = 0;
};


struct DenseLayer : NetworkLayer
{
public:
    DenseLayer(int size, int input_size);
    ~DenseLayer();
     float* FeedForward(float* input_activations, int size) override;

     void SetWeights(float* data, int size);
     void SetBiases(float* data, int size);
    

    float GetWeight(int src_neuron, int dst_neuron) override;
    std::vector<int> GetDimensions() override;
    LayerActivation GetLayerActivation() override;

    
    
private:
    int m_InputSize;
    int m_LayerSize;
    std::vector<float> m_Weights;
    std::vector<float> m_Activation;
    std::vector<float> z;
};


struct BackPropResult
{
    std::vector<std::vector<std::vector<float>>> gradient;
    std::vector<std::vector<float>> bias_gradient;
};

class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<std::unique_ptr<NetworkLayer>>& layers);
    ~NeuralNetwork();


    void SGD(std::vector<std::vector<float>> &training_data, int epochs, int mini_batch_size, int eta, const std::vector<std::vector<float>> &test_data);
    void UpdateMiniBatch(const std::vector<std::pair<int, std::vector<float>>> &mini_batch, int eta);   
    std::vector<LayerActivation> FeedforwardTrain(std::vector<float>& input_instance);


    int Inference(std::vector<float>& input_instance);
    std::vector<float> Evaluate(const std::vector<std::vector<float>> &test_data);
private:

    std::vector<std::unique_ptr<NetworkLayer>> m_Layers;
    BackPropResult Backpropagation(std::vector<float>& input_instance, int label);
    std::vector<float> BaseCostDerivative(LayerActivation output_layer_activation, int label);
    std::vector<std::vector<std::vector<float>>> AllocGradientStorage();
    std::vector<std::vector<float>> AllocBiasGradientStorage();
    float GetWeight(int layer, int dst_neuron, int src_neuron);
};


#endif