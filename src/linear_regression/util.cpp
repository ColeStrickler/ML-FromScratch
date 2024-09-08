#include "util.h"





float util::Random()
{
    // Create a random number generator
    std::random_device rd;   // obtain a random number from hardware
    std::mt19937 gen(rd());  // seed the generator
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);  // define the range

    // Generate a random float between 0 and 1
    return dis(gen);
}
