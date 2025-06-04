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


inline float sigmoid(float val)
{
    return 1.0f/(1.0 + std::exp(-val));
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
};



class NetworkLayer
{
public:
    LayerType m_Type;
    virtual float* FeedForward(float* input_activations, int size) = 0; // pure virtual 
    virtual ~NetworkLayer() = default;  // Always add virtual destructor for base classes
    virtual LayerActivation GetLayerActivation() = 0; // pure virtual 
    virtual std::vector<int> GetDimensions() = 0;
};


struct DenseLayer : NetworkLayer
{
public:
    DenseLayer(int size, int input_size);
    ~DenseLayer();
     float* FeedForward(float* input_activations, int size) override;

     void SetWeights(float* data, int size);
     void SetBiases(float* data, int size);
    
    std::vector<int> GetDimensions() override;
     LayerActivation GetLayerActivation() override;
    
private:
    int m_InputSize;
    int m_LayerSize;
    float* m_Weights;
    float* m_Activation;
};






class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<std::unique_ptr<NetworkLayer>>& layers);
    ~NeuralNetwork();


    void SGD(std::vector<std::vector<float>> &training_data, int epochs, int mini_batch_size, int eta, const std::vector<std::vector<float>> &test_data);
    void UpdateMiniBatch(const std::vector<std::vector<float>>& mini_batch, int eta);    
    std::vector<LayerActivation> FeedforwardTrain(std::vector<float>& input_instance);


    int Inference(std::vector<float>& input_instance);
    std::vector<float> Evaluate(const std::vector<std::vector<float>> &test_data);
private:
    std::vector<std::unique_ptr<NetworkLayer>> m_Layers;
    void Backpropagation(std::vector<float>& input_instance);
    

};


#endif