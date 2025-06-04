
#include <stdlib.h>
#include "model.h"
#include "data.h"


int main(int argc, char** argv)
{
    
    std::vector<std::vector<double>> train_in(data_in.begin() + 20, data_in.end());
    std::vector<double> train_out(data_out.begin() + 20, data_out.end());


    auto model = LinearRegressionModel(train_in, train_out, 0.000007396f);
    //model.SetBatchSize(0);
    model.Train(0.000000001f, 1000000);



    for (int i = 0; i < 20; i++)
    {
        auto prediction = model.Predict(data_in[i]);
        printf("Prediction: %.3f\n", prediction);
        printf("Actual: %.3f, Error: %.3f\n", train_out[i], abs(prediction - train_out[i])/train_out[i] * 100.0f);
    }
    


}