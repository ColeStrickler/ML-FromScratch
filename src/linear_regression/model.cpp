#include "model.h"

LinearRegressionModel::LinearRegressionModel(std::vector<std::vector<double>> trainingDataIn, std::vector<double> trainingDataOut, double learningRate)
{

    assert(trainingDataIn.size() == trainingDataOut.size());
    m_TrainingDataIn = trainingDataIn;
    m_TrainingDataOut = trainingDataOut;
    m_LearningRate = learningRate;
    m_BatchSize = trainingDataIn.size();
    for (int i = 0; i < trainingDataIn[0].size(); i++)
        m_Parameters.push_back(util::Random()*10.0f); // initialize parameters randomly'
}

LinearRegressionModel::~LinearRegressionModel()
{
}

void LinearRegressionModel::SetBatchSize(int batch_size)
{
    m_BatchSize = batch_size;
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
bool LinearRegressionModel::Train(double error_margin, int max_epochs)
{
    int epoch = 0;


    while (epoch < max_epochs)
    {
        double avg_cost = 0.0f;
        std::vector<double> updatedParams;
        for (int j = 0; j < m_Parameters.size(); j++)
        {
            /*
                Updating of parameter j
                parameter_j^(i+1) = parameter_j^(i) - (learning_rate)d/dparam_j(CostFunction(training_set(i)))
            */
           
           
           
           auto training_batch = m_BatchSize != m_TrainingDataIn.size() ? GenTrainingBatch() : m_TrainingDataIn;
           double total_cost = 0.0f;
           for (int i = 0; i < training_batch.size(); i++)
           {
                double cost = (Predict(training_batch[i], m_Parameters) - m_TrainingDataOut[i]) * training_batch[i][j];
                total_cost += cost;
           }
           
           double param_j_i1 = m_Parameters[j] - m_LearningRate*(total_cost);
           //printf("ParamJ %.5f, total_cost: %.5f\n", param_j_i1, total_cost);
           updatedParams.push_back(param_j_i1);
           avg_cost += total_cost;   
        }

        m_Parameters.clear();
        m_Parameters = updatedParams;
        //for (int i = 0; i < m_Parameters.size(); i++)
        //    printf("Parameter(%d): %.3f\n", i, m_Parameters[i]);
        //printf("Epochs: %d AvgCost: %.5f\n", epoch, abs(avg_cost / m_Parameters.size()));
        if (epoch%100 == 0)
            printf("Epoch %d/%d Average Cost: %.4f\n", epoch, max_epochs, avg_cost/m_Parameters.size());
        //if (abs(avg_cost / m_Parameters.size()) <= error_margin)
        //{
        //    printf("Epochs: %d AvgCost: %.5f\n", epoch, abs(avg_cost / m_Parameters.size()));
        //    return true; 
        //}
                 
        epoch++;
    }
    printf("Epochs: %d\n", epoch);
    return false;
}

double LinearRegressionModel::Predict(std::vector<double> data)
{
    //for (int i = 0; i < m_Parameters.size(); i++)
    //    printf("Parameter(%d): %.3f\n", i, m_Parameters[i]);
    assert(data.size() == m_Parameters.size());
    double sum = 0.0f;
    for (int i = 0; i < data.size(); i++)
        sum += (data[i]*m_Parameters[i]);
    return sum;
}

double LinearRegressionModel::Predict(std::vector<double> data, std::vector<double> params)
{
    assert(data.size() == params.size());

    double sum = 0.0f;
    for (int i = 0; i < data.size(); i++)
        sum += (data[i]*params[i]);
    return sum;
}

std::vector<std::vector<double>> LinearRegressionModel::GenTrainingBatch()
{
    std::vector<std::vector<double>> training_batch;
    for (int x = 0; x < m_BatchSize; x++)
    {
        training_batch.push_back(m_TrainingDataIn[int(util::Random()*(m_TrainingDataIn.size() - 1))]);
    }
    return training_batch;
}
