
#include <stdlib.h>
#include "model.h"


int main(int argc, char** argv)
{
    std::vector<std::vector<float>> train_in = {
        {2, 4},
        {6, 12},
        {9, 3},
        {11, 1},
        {6, 7}
    };
    std::vector<float> train_out = {
        12,
        36,
        24,
        24,
        26
    };


    auto model = LinearRegressionModel(train_in, train_out, 0.0002f);

    model.Train(0.0000000001f, 100000000);

    printf("Prediction: %.3f\n", model.Predict({4.0f, 4.0f}));


}