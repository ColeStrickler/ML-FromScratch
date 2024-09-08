#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <string>
#include <cassert>
#include "util.h"
class LinearRegressionModel {
public:
    LinearRegressionModel(std::vector<std::vector<float>> trainingDataIn, std::vector<float> trainingDataOut, float learningRate);
    ~LinearRegressionModel();

    void ClearTrainingData();
    bool Train(float error_margin, int max_epochs);
    float Predict(std::vector<float> data);

    float Predict(std::vector<float> data, std::vector<float> params);
private:
    float m_LearningRate;
    std::vector<float> m_Parameters;
    std::vector<float> m_TrainingDataOut;
    std::vector<std::vector<float>> m_TrainingDataIn;



};







#endif