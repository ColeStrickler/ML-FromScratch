#include "model.h"

LinearRegressionModel::LinearRegressionModel(std::vector<std::vector<float>> trainingDataIn, std::vector<float> trainingDataOut, float learningRate)
{

    assert(trainingDataIn.size() == trainingDataOut.size());
    m_TrainingDataIn = trainingDataIn;
    m_TrainingDataOut = trainingDataOut;
    m_LearningRate = learningRate;
    for (int i = 0; i < trainingDataIn[0].size(); i++)
        m_Parameters.push_back(util::Random()*10.0f); // initialize parameters randomly'
    for (int i = 0; i < m_Parameters.size(); i++)
            printf("Parameter(%d): %.3f\n", i, m_Parameters[i]);
}

LinearRegressionModel::~LinearRegressionModel()
{
}

void LinearRegressionModel::ClearTrainingData()
{
    m_TrainingDataIn.clear();
    m_TrainingDataIn.shrink_to_fit();
    m_TrainingDataOut.clear();
    m_TrainingDataOut.shrink_to_fit();
}


/*
    Return true if we were able to achieve the error margin, return false if not

    We will return early if we achieve the error margin before max_epochs
*/
bool LinearRegressionModel::Train(float error_margin, int max_epochs)
{
    int epoch = 0;


    while (epoch < max_epochs)
    {


        float avg_cost = 0.0f;
        std::vector<float> updatedParams;
        for (int j = 0; j < m_Parameters.size(); j++)
        {
            /*
                Updating of parameter j
                parameter_j^(i+1) = parameter_j^(i) - (learning_rate)d/dparam_j(CostFunction(training_set(i)))
            */
           
           float total_cost = 0.0f;
           for (int i = 0; i < m_TrainingDataIn.size(); i++)
           {
                float cost = (Predict(m_TrainingDataIn[i], m_Parameters) - m_TrainingDataOut[i]) * m_TrainingDataIn[i][j];
                total_cost += cost;;
           }
           
           float param_j_i1 = m_Parameters[j] - m_LearningRate*total_cost;
           updatedParams.push_back(param_j_i1);
           avg_cost += total_cost;   
        }
        m_Parameters.clear();
        m_Parameters = updatedParams;
        //for (int i = 0; i < m_Parameters.size(); i++)
        //    printf("Parameter(%d): %.3f\n", i, m_Parameters[i]);
        //printf("Average Cost: %.2f\n", avg_cost/m_Parameters.size());
        if (abs(avg_cost / m_Parameters.size()) <= error_margin)
        {
            printf("Epochs: %d\n", epoch);
            return true; 
        }
                 
        epoch++;
    }
    printf("Epochs: %d\n", epoch);
    return false;
}

float LinearRegressionModel::Predict(std::vector<float> data)
{
    //for (int i = 0; i < m_Parameters.size(); i++)
    //    printf("Parameter(%d): %.3f\n", i, m_Parameters[i]);
    assert(data.size() == m_Parameters.size());
    float sum = 0.0f;
    for (int i = 0; i < data.size(); i++)
        sum += (data[i]*m_Parameters[i]);
    return sum;
}

float LinearRegressionModel::Predict(std::vector<float> data, std::vector<float> params)
{
    assert(data.size() == params.size());

    float sum = 0.0f;
    for (int i = 0; i < data.size(); i++)
        sum += (data[i]*params[i]);
    return sum;
}
