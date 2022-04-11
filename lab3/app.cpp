#include "NeuralNetwork.hpp"

#define PATH_TO_LAYERS "../layers_files/"

int main() {

    NeuralNetwork network(1);
    vector<vector<float>> inputSeries = {
        {0.5, 0.75, 0.1}, 
        {0.1, 0.3, 0.7}, 
        {0.2, 0.1, 0.6}, 
        {0.8, 0.9, 0.2}
    };

    network.activationFunctions.push_back(relu);

    // vector<vector<float>> expectedOutputSeries = {
    //     {0.1, 1.0, 0.1, 0.0, -0.1}, 
    //     {0.5, 0.2, -0.5, 0.3, 0.7}, 
    //     {0.1, 0.3, 0.2, 0.9, 0.1}, 
    //     {0.7, 0.6, 0.2, -0.1, 0.8}
    // };

    float alpha = 0.01;

    string filename = "layers_example.json";
    network.loadWeights(PATH_TO_LAYERS + filename);


    return EXIT_SUCCESS;
}