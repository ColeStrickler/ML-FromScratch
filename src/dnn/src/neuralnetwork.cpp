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
    //printf("a.size %d, b.size %d\n", a.size(), b.size());
    assert(a.size() == b.size());
    std::vector<float> ret;

    for (int i = 0; i < a.size(); i++)
    {
        ret.push_back(a[i]*b[i]);
        //printf("Hadamard %.2f*%.2f\n", a[i], b[i]);
    }
    return ret;
}

DenseLayer::DenseLayer(int size, int input_size) : m_LayerSize(size), m_InputSize(input_size)
{
    m_Weights = std::vector<float>(size * (input_size + 1)); // each neuron needs a weight for each input + bias
    m_Activation = std::vector<float>(size);                 // an output from each neuron
    z = std::vector<float>(size);
    //printf("DenseLayer Size (%d, %d)\n", input_size, size);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < input_size + 1; j++)
        {
            m_Weights[i * (m_InputSize + 1) + j] = RandomFloatGenerator::xavier(input_size, size); // randomize weights
                                                                                // printf("DenseLayer Size (%d, %d)\n", i, j);
        }
}

float DenseLayer::GetWeight(int dst_neuron, int src_neuron)
{
    //printf("dst_neuron %d < m_LayerSize %d\n", dst_neuron, m_LayerSize);
    assert(dst_neuron >= 0 && dst_neuron < m_LayerSize);
    //printf("src_neuron %d <= m_InputSize %d\n", src_neuron, m_InputSize);
    assert(src_neuron >= 0 && src_neuron <= m_InputSize); // <= for bias access
    return m_Weights[dst_neuron*(m_InputSize+1) + src_neuron];
}

void DenseLayer::SetWeight(int dst_neuron, int src_neuron, float value)
{
    assert(dst_neuron >= 0 && dst_neuron < m_LayerSize);
    assert(src_neuron >= 0 && src_neuron <= m_InputSize); // <= for bias access
    m_Weights[dst_neuron*(m_InputSize+1) + src_neuron] = value;
}

DenseLayer::~DenseLayer()
{
}

float *DenseLayer::FeedForward(float *input_activations, int size)
{
    if (size != m_InputSize)
        printf("%d,%d\n", size, m_InputSize);
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

void NeuralNetwork::SGD(std::vector<std::pair<int, std::vector<float>>> &training_data, int epochs, int mini_batch_size, float learning_rate,
                        const std::vector<std::pair<int, std::vector<float>>> &test_data)
{

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        printf("Epoch %d/%d\n", epoch, epochs);
        std::random_device rd;
        std::mt19937 gen(rd()); // Random engine
        std::shuffle(training_data.begin(), training_data.end(), gen);

        for (int j = 0; j < training_data.size(); j += mini_batch_size)
        {
            std::vector<std::pair<int, std::vector<float>>> mini_batch(training_data.begin() + j, training_data.begin() + j + mini_batch_size);

            UpdateMiniBatch(mini_batch, learning_rate);
        }
        //printf("Done train cycle\n");

        int correct = Evaluate(test_data);
        printf("Test data: %d/%d\n", correct, test_data.size());
    }
}

int NeuralNetwork::Inference(std::vector<float> &input_instance)
{
    float *activation = input_instance.data();
    int activation_size = input_instance.size();

    assert(input_instance.size());


    auto output_layer = m_Layers[m_Layers.size()-1].get();
    for (auto &layer : m_Layers)
    {
       // printf("ff\n");
        layer->FeedForward(activation, activation_size);
        auto layer_activation = layer->GetLayerActivation();
        activation = layer_activation.activations;
        activation_size = layer_activation.size;
    }

    return getGuess(output_layer->GetLayerActivation());
}

int NeuralNetwork::Evaluate(const std::vector<std::pair<int, std::vector<float>>> &test_data)
{
    int correct = 0;
    for (auto& instance: test_data)
    {
        auto inst = instance.second;
        auto label = instance.first;
        auto guess = Inference(inst);
        correct += (guess == label ? 1 : 0);
    }
    return correct;
}

BackPropResult NeuralNetwork::Backpropagation(std::vector<float> &input_instance, int label)
{
    //printf("Backpropagation()\n");
    BackPropResult ret;
    auto nabla_w = AllocGradientStorage();
    auto nabla_b = AllocBiasGradientStorage();
    if (!input_instance.size())
        return ret;
    /*
        We need to rethink this. Want to access like matrix[L][k][j]
    */


    auto activations = FeedforwardTrain(input_instance);
    auto output_layer_activation = activations[activations.size()-1];

    std::vector<float> x;
    for (int i = 0; i < output_layer_activation.size; i++)
    {
        x.push_back(sigmoid_prime(output_layer_activation.z[i]));
       // printf("%.2f\n", output_layer_activation.z[i]);
    }
        


    //auto x = applyToEachManual(output_layer_activation.z, output_layer_activation.size, sigmoid_prime);
    auto cost_derivative = BaseCostDerivative(output_layer_activation, label);
    std::vector<float> delta_L = Hadamard(cost_derivative,\
     x);

    int L = m_Layers.size()-1;



    nabla_b[L] = delta_L;

    std::vector<float> last;
    for (int j = 0; j < delta_L.size(); j++) // activation of neuron_L-1_k to neuron_L_j
    {
        auto activation = activations[L-1];
        float tmp = 0.0f; // I think this is right
        for (int k = 0; k < activation.size; k++)
        {
            assert(j < delta_L.size());
            assert(L < nabla_w.size());
            assert(j < nabla_w[L].size());
           // printf("%d,%d, activation size %d\n", k, nabla_w[L][j].size(), activation.size);
            assert(k < nabla_w[L][j].size());
            assert(k < activation.size);
            nabla_w[L][j][k] = delta_L[j]*activation.activations[k];
            //printf("nabla_w[%d][%d][%d] %.4f\n", L, j, k, nabla_w[L][j][k]);
            //printf("delta_L[%d] %.4f\n", j, delta_L[j]);
            //printf("costD[%d] %.4f\n", j, cost_derivative[j]);
            //printf("x[%d] %.4f\n", j, x[j]);
        }   
    }
    L--;


    while (L > 0)
    {
      // printf("back prop %d\n", L);
        float* z = activations[L].z;
        int z_size = activations[L].size;
         
        auto sigmoid_prime_z = applyToEachManual(z, z_size, sigmoid_prime);
        
        std::vector<float> prev_delta_L(delta_L);
       
        delta_L = std::vector<float>(z_size, 0.0f); // delta value for each neuron in layer
        
        for (int j = 0; j < delta_L.size(); j++)
        {
            for (int k = 0; k < prev_delta_L.size(); k++)
            {
                /*
                    delta_L_j = SUM[(w_L+1_k_j)(sigma_prime(z_L_j))(delta_L+1_k)
                */
               // printf("%d, %d\n", delta_L.size(), j);
                auto res = GetWeight(L+1, k, j)*sigmoid_prime(z[j])*prev_delta_L[k];
                //printf("res %.4f\n", res);
                delta_L[j] +=  res;// need to get w_L+1_k_j
                //printf("delta_L[%d] %.4f\n", j, delta_L[j]);
                //printf("z[%d] %.2f\n", j, z[j]);
                //printf("prev_delta_L[%d] %.4f\n", k, prev_delta_L[k]);
                break;
            }
            
        }
        
        nabla_b[L] = delta_L;
        for (int j = 0; j < delta_L.size(); j++) // activation of neuron_L-1_k to neuron_L_j
        {
            auto activation = activations[L-1];
            float tmp = 0.0f; // I think this is right
            for (int k = 0; k < activation.size; k++)
            {
                //printf("a\n");
                nabla_w[L][j][k] = delta_L[j]*activation.activations[k];\
                //if (nabla_w[L][j][k] > 0.03f)
                //    printf("nabla_w[%d][%d][%d] %.7f\n", L, j, k, nabla_w[L][j][k]);
                //exit(-1);
                //printf("b\n");
                //printf("back prop %d\n", L);
            }   
        }
        //printf("back propx %d\n", L);
        L--;
    }
    
    
    ret.gradient = nabla_w;
    ret.bias_gradient = nabla_b;
   // printf("Done!\n");
    return ret;
}


void NeuralNetwork::UpdateMiniBatch(const std::vector<std::pair<int, std::vector<float>>> &mini_batch, float learning_rate)
{
 //   printf("UpdateMiniBatch()\n");

    auto nabla_w = AllocGradientStorage();
    auto nabla_b = AllocBiasGradientStorage();
    int mini_batch_size = mini_batch.size();

    for (auto& instance: mini_batch)
    {
       // printf("MiniBatch\n");
        auto inst = instance.second;
        auto label = instance.first;
        if (!inst.size())
            continue; // safe guards

        // we need there to be labels in the instance --> fix
        BackPropResult res = Backpropagation(inst, label);
       // printf("Finished Backprop\n");
        for (int l = 0; l < nabla_w.size(); l++)
            for (int j = 0; j < nabla_w[l].size(); j++)
                for (int k = 0; k < nabla_w[l][j].size(); k++)
                {
                    //printf("nabla_w %d, %d, %d\n", l, j, k);
                    nabla_w[l][j][k] += res.gradient[l][j][k];
                    //printf("res.gradient[%d][%d][%d]: %.4f\n", l,j,k, res.gradient[l][j][k]);
                }

        
        for (int l = 0; l < nabla_b.size(); l++)
            for (int j = 0; j < nabla_b[l].size(); j++)
                {
                   // printf("here nabla_b %d, %d\n", l, j);
                    nabla_b[l][j] += res.bias_gradient[l][j];
                   // printf("res.bias_gradient[%d][%d][%d]: %.4f\n", l,j, res.bias_gradient[l][j]);
                  //  printf("here nabla_b %d, %d\n", l, j);
                }
        // I think all we need to do now is sum the nablas, and then update the weights. Also add labels
    }

   
    for (int l = 0; l < nabla_w.size(); l++)
        for (int j = 0; j < nabla_w[l].size(); j++)
            for (int k = 0; k < nabla_w[l][j].size(); k++)
            {
               // printf("nabla_w %d %d\n", nabla_w[l].size(), j);
                SetWeight(l, j, k, GetWeight(l, j, k)-(learning_rate/mini_batch_size)*nabla_w[l][j][k]);
            }
                

    for (int l = 0; l < nabla_b.size(); l++)
    {
        auto dim = m_Layers[l]->GetDimensions();
        int k_value = dim[0]; // input size = # weights
        for (int j = 0; j < nabla_b[l].size(); j++)
        {
            //printf("nabla_b %d %d\n", nabla_b[l].size(), nabla_b.size());
            SetWeight(l, j, k_value, GetWeight(l, j, k_value)-(learning_rate/mini_batch_size)*nabla_b[l][j]);
        }
    }
  //  printf("finished MiniBatch\n");
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
    assert(layer >= 0 && layer < m_Layers.size()); //
    auto sel_layer = m_Layers[layer].get();
    return sel_layer->GetWeight(dst_neuron, src_neuron);
}


void NeuralNetwork::SetWeight(int layer, int dst_neuron, int src_neuron, float value)
{
    assert(layer >= 0 && layer < m_Layers.size());
    auto sel_layer = m_Layers[layer].get();

    return sel_layer->SetWeight(dst_neuron, src_neuron, value);
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
        std::vector<std::vector<float>>layer_gradient(num_neurons, std::vector<float>(input_size));
        ret.push_back(layer_gradient);
    }


    return ret;
}

std::vector<LayerActivation> NeuralNetwork::FeedforwardTrain(std::vector<float> &input_instance)
{

    std::vector<LayerActivation> activations_all;
    float *activation = input_instance.data();
    int activation_size = input_instance.size();
    assert(input_instance.size());
    for (auto &layer : m_Layers)
    {
        //printf("fft\n");
        layer->FeedForward(activation, activation_size);
        auto layer_activation = layer->GetLayerActivation();
        activation = layer_activation.activations;
        activation_size = layer_activation.size;
        activations_all.push_back(layer_activation);
    }

    return activations_all;
}
