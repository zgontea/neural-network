#include "NeuralNetwork.hpp"

#define PATH_TO_LAYERS "../layers_files/"

int main() {

    NeuralNetwork network(3);
    vector<vector<float>> inputSeries = {
        {0.5, 0.75, 0.1}, 
        {0.1, 0.3, 0.7}, 
        {0.2, 0.1, 0.6}, 
        {0.8, 0.9, 0.2}
    };

    network.activationFunctions.push_back(relu);
    network.activationFunctions.push_back(NULL);

    vector<vector<float>> expectedOutputSeries = {
        {0.1, 1.0, 0.1}, 
        {0.5, 0.2, -0.5}, 
        {0.1, 0.3, 0.2}, 
        {0.7, 0.6, 0.2}
    };

    float alpha = 0.01;

    string filename = "layers_example.json";
    network.loadWeights(PATH_TO_LAYERS + filename);

    cout << "\nLab 3.4\n";

    for(vector<float> input : inputSeries) {
        vector<float> output = network.predict(input);
        for(float value : output) {
            cout << value << '\n';
        }
        cout << '\n';
    }

    cout << "\nLab 3.5\n";

    vector<vector<float>> outputSeries = network.teachForSeries(50, inputSeries, alpha, expectedOutputSeries);

    for(vector<float> output : outputSeries) {
        for(float value : output) {
            cout << value << '\n';
        }
        cout << '\n';
    }

    return EXIT_SUCCESS;
}