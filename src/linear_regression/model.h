#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <string>
#include <cassert>
#include "util.h"
class LinearRegressionModel {
public:
    LinearRegressionModel(std::vector<std::vector<double>> trainingDataIn, std::vector<double> trainingDataOut, double learningRate);
    ~LinearRegressionModel();
    void SetBatchSize(int batch_size);
    void ClearTrainingData();
    bool Train(double error_margin, int max_epochs);
    double Predict(std::vector<double> data);

    double Predict(std::vector<double> data, std::vector<double> params);
private:
    std::vector<std::vector<double>> GenTrainingBatch();


    int m_BatchSize;
    double m_LearningRate;
    std::vector<double> m_Parameters;
    std::vector<double> m_TrainingDataOut;
    std::vector<std::vector<double>> m_TrainingDataIn;



};







#endif