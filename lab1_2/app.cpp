#include "NeuralNetwork.hpp"

#define PATH_TO_LAYERS "../layers_files/"

int main() {

    vector<float> input;
    int inputCount;

    cout << "Enter input count:\n";
    cin >> inputCount;

    if(cin.fail() || inputCount < 1) {
        cout << "Incorrect input\n";
        return 1;
    }

    input.resize(inputCount);
    
    for(int i = 0; i < inputCount; i++) {
        cout << "Enter input value " << i << ": ";
        cin >> input[i];

        if(cin.fail()) {
        cout << "Incorrect input\n";
        return 2;
        }
    }

    NeuralNetwork network(inputCount);

    vector<float> result;
    string filename;
    int outputCount;

    int epochCount;
    float learningRatio;
    vector<float> expectedOutput;
    int outputSize;

    vector<vector<float>> inputSeries = {
        {0.5, 0.75, 0.1}, 
        {0.1, 0.3, 0.7}, 
        {0.2, 0.1, 0.6}, 
        {0.8, 0.9, 0.2}
    };

    vector<vector<float>> expectedOutputSeries = {
        {0.1, 1.0, 0.1, 0.0, -0.1}, 
        {0.5, 0.2, -0.5, 0.3, 0.7}, 
        {0.1, 0.3, 0.2, 0.9, 0.1}, 
        {0.7, 0.6, 0.2, -0.1, 0.8}
    };

    vector<vector<float>> inputColors;
    vector<vector<float>> expectedOutputColors;

    fstream colorsFile;
    colorsFile.open("../training_colors.txt", ios::in);

    string line;
    while(getline(colorsFile, line)) {
        vector<float> data;
        size_t pos = 0;
        string token;
        while((pos = line.find(" ")) != std::string::npos) {
            token = line.substr(0, pos);
            // std::cout << token << std::endl;
            data.push_back(stof(token));
            line.erase(0, pos + 1);
        }
        inputColors.push_back(data);
        vector<float> expectedData = {0, 0, 0, 0};
        expectedData[stof(line) - 1] = 1;
        expectedOutputColors.push_back(expectedData);
    }

    vector<vector<float>> inputColorsTest;
    vector<float> expectedOutputColorsTest;

    int correct = 0;
    float percentage;

    fstream colorsFileTest;
    colorsFileTest.open("../test_colors.txt", ios::in);

    while(getline(colorsFileTest, line)) {
        vector<float> data;
        size_t pos = 0;
        string token;
        while((pos = line.find(" ")) != std::string::npos) {
            token = line.substr(0, pos);
            // std::cout << token << std::endl;
            data.push_back(stof(token));
            line.erase(0, pos + 1);
        }
        inputColorsTest.push_back(data);
        expectedOutputColorsTest.push_back(stof(line));
    }

    vector<vector<float>> resultSeries;

    cout << "Available options:\n" << "1 - add layer\n"
        << "2 - predict\n" << "3 - load file\n" << "4 - teach\n"
        << "5 - teach for series\n" << "6 - colors test\n" << "7 - close\n";

    bool loop = true;
    while(loop) {
        cout << "Choose option: ";
        int op = 6;

        while(getchar() != '\n');

        if(scanf("%d", &op) != 1) {
            cout << "Incorrect input\n";
            continue;
        }

        switch(op) {
            case 1:
                int neurons;
                cout << "Enter neurons count: ";

                while(getchar() != '\n');

                if(scanf("%d", &neurons) != 1) {
                    cout << "Incorrect input\n";
                    continue;
                }

                if(neurons < 1) {
                    cout << "Incorrect input\n";
                    continue;
                }

                network.addLayer(neurons);
                break;

            case 2:
                result = network.predict(input);

                cout << "Result:\n";
                outputCount = 0;
                for(float value : result) {
                    cout << "Output " << outputCount << ": " << value << '\n';
                    outputCount++;
                }
                break;

            case 3:
                cout << "Enter filename: ";
                cin >> filename;

                network.loadWeights(PATH_TO_LAYERS + filename);
                break;

            case 4:
                if(network.getLayersCount() != 1) {
                    cout << "Function not available for multiple layers\n";
                    break;
                }

                cout << "Enter epoch count: ";
                cin >> epochCount;

                cout << "Enter learning ratio: ";
                cin >> learningRatio;

                outputSize = network.getNeuronsCountByLayer(0);
                cout << "Enter expected output (" << outputSize << " values): ";
                for(int i = 0; i < outputSize; i++) {
                    float value;
                    cin >> value;
                    expectedOutput.push_back(value);
                }

                result = network.teach(epochCount, input, learningRatio, expectedOutput);

                cout << "Result:\n";
                outputCount = 0;
                for(float value : result) {
                    cout << "Output " << outputCount << ": " << value << '\n';
                    outputCount++;
                }
                break;

            case 5:
                if(network.getLayersCount() != 1) {
                    cout << "Function not available for multiple layers\n";
                    break;
                }

                cout << "Enter epoch count: ";
                cin >> epochCount;

                cout << "Enter learning ratio: ";
                cin >> learningRatio;

                resultSeries = network.teachForSeries(epochCount, inputSeries, learningRatio, expectedOutputSeries);
                // resultSeries = network.teachForSeries(epochCount, inputColors, learningRatio, expectedOutputColors);

                cout << "Result:\n";
                for(int s = 0; s < (int)inputSeries.size(); s++) {
                    cout << "Series " << s << "\n";
                    outputCount = 0;
                    for(float value : resultSeries[s]) {
                        cout << "Output " << outputCount << ": " << value << '\n';
                        outputCount++;
                    }
                }
                break;

            case 6:
                if(network.getLayersCount() != 1) {
                    cout << "Function not available for multiple layers\n";
                    break;
                }

                cout << "Enter epoch count: ";
                cin >> epochCount;

                cout << "Enter learning ratio: ";
                cin >> learningRatio;

                // resultSeries = network.teachForSeries(epochCount, inputSeries, learningRatio, expectedOutputSeries);
                resultSeries = network.teachForSeries(epochCount, inputColors, learningRatio, expectedOutputColors);

                cout << "Result:\n";
                for(int s = 0; s < (int)inputColors.size(); s++) {
                    cout << "Series " << s << "\n";
                    outputCount = 0;
                    for(float value : resultSeries[s]) {
                        cout << "Output " << outputCount << ": " << value << '\n';
                        outputCount++;
                    }
                }

                correct = 0;
                for(int i = 0; i < (int)expectedOutputColorsTest.size(); i++) {
                    vector<float> resultColor = network.predict(inputColorsTest[i]);
                    float max = resultColor[0];
                    int maxIndex = 0;
                    for(int m = 0; m < (int)resultColor.size(); m++) {
                        if(resultColor[m] > max) {
                            max = resultColor[m];
                            maxIndex = m;
                        }
                    }

                    if(maxIndex + 1 == expectedOutputColorsTest[i]) {
                        correct++;
                    }                  
                }
                cout << "Correct: " << correct << "\n";
                cout << "All: " << expectedOutputColorsTest.size() << "\n";
                percentage = correct / (float)expectedOutputColorsTest.size();
                cout << "Percentage: " << percentage * 100 << "%\n";
                break;

            case 7:
                loop = false;
                break;

            default:
                cout << "Incorrect option\n";
                break;
        }
    }
  
    return 0;
}