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

    vector<vector<float>> layersOutput;

    for(vector<float> input : inputSeries) {
        vector<float> output = network.predict(input, layersOutput);
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

    cout << "\nLab 3.6\n";

    FILE *imagesTrain = fopen("../images/train-images.idx3-ubyte", "rb");
    FILE *labelsTrain = fopen("../images/train-images.idx3-ubyte", "rb");

    vector<vector<float>> inputImages;
    inputImages.resize(60000);
    for(int i = 0; i < 60000; i++) {
        inputImages[i].resize(28 * 28);
    }

    vector<float> inputLabels;
    inputLabels.resize(60000);

    fseek(imagesTrain, 16, SEEK_SET);
    fseek(labelsTrain, 8, SEEK_SET);

    uint8_t byte;
    for(int i = 0; i < 10; i++) {
        fread(&byte, 1, 1, labelsTrain);
        inputLabels[i] = byte;
        for(int y = 0; y < 28; y++) {
            for(int x = 0; x < 28; x++) {
                fread(&byte, 1, 1, imagesTrain);
                inputImages[i][(y + 1) * x] = byte / 255.0;
                // cout << inputImages[i][(y + 1) * x] << ' ';
            }
            // cout << '\n';
        }
    }

    NeuralNetwork neuralImage(784);
    neuralImage.addLayer(40);
    neuralImage.addLayer(10);

    neuralImage.activationFunctions.push_back(relu);
    neuralImage.activationFunctions.push_back(NULL);

    vector<vector<float>> expectedImageOutput;
    expectedImageOutput.resize(60000);
    for(int i = 0; i < 10; i++) {
        expectedImageOutput[i].resize(10);
        for(int x = 0; x < 10; x++) {
            expectedImageOutput[i][x] = 0;
        }
        cout << (int)inputLabels[i];
        expectedImageOutput[i][inputLabels[i]] = 1;
    }

    // vector<vector<float>> imagesResult = neuralImage.teachForSeries(1, inputImages, 0.01, expectedImageOutput);

    // for(int i = 0; i < 10; i++) {
    //     cout << imagesResult[0][i] << '\n';
    // }

    return EXIT_SUCCESS;
}