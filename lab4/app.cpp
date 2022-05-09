#include "NeuralNetwork.hpp"

#define PATH_TO_LAYERS "../layers_files/"

int main()
{

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
    FILE *labelsTrain = fopen("../images/train-labels.idx1-ubyte", "rb");

    vector<vector<float>> inputImages;
    inputImages.resize(60000);
    for (int i = 0; i < 60000; i++)
    {
        inputImages[i].resize(28 * 28);
    }

    fseek(imagesTrain, 16, SEEK_SET);
    fseek(labelsTrain, 8, SEEK_SET);

    uint8_t byte;
    for (int i = 0; i < 60000; i++)
    {
        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
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
    neuralImage.activationFunctions.push_back(relu);

    vector<vector<float>> expectedImageOutput;
    expectedImageOutput.resize(60000);
    for (int i = 0; i < 60000; i++)
    {
        expectedImageOutput[i].resize(10);
        for (int x = 0; x < 10; x++)
        {
            expectedImageOutput[i][x] = 0;
        }
        fread(&byte, 1, 1, labelsTrain);
        expectedImageOutput[i][(int)byte] = 1;
    }

    vector<vector<float>> lo;
    vector<float> outputS = neuralImage.predict(inputImages[0], lo);
    
    vector<vector<float>> imagesResult;
    imagesResult = neuralImage.teachForSeries(1, inputImages, 0.01, expectedImageOutput);

    FILE *imagesTest = fopen("../images/t10k-images.idx3-ubyte", "rb");
    FILE *labelsTest = fopen("../images/t10k-labels.idx1-ubyte", "rb");

    inputImages.resize(10000);
    for (int i = 0; i < 10000; i++)
    {
        inputImages[i].resize(28 * 28);
    }

    fseek(imagesTest, 16, SEEK_SET);
    fseek(labelsTest, 8, SEEK_SET);

    for (int i = 0; i < 10000; i++)
    {
        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                fread(&byte, 1, 1, imagesTest);
                inputImages[i][(y + 1) * x] = byte / 255.0;
                // cout << inputImages[i][(y + 1) * x] << ' ';
            }
            // cout << '\n';
        }
    }

    expectedImageOutput.resize(10000);
    for (int i = 0; i < 10000; i++)
    {
        for (int x = 0; x < 10; x++)
        {
            expectedImageOutput[i][x] = 0;
        }
        fread(&byte, 1, 1, labelsTest);
        expectedImageOutput[i][(int)byte] = 1;
    }

    imagesResult.resize(10000);
    for (int i = 0; i < 10000; i++)
    {
        imagesResult[i] = neuralImage.predict(inputImages[i], lo);
    }

    int correct = 0;
    for (int i = 0; i < 10000; i++)
    {
        float max = imagesResult[i][0];
        int indexMax = 0;
        for (int x = 0; x < 10; x++)
        {
            if (imagesResult[i][x] > max)
            {
                max = imagesResult[i][x];
                indexMax = x;
            }
        }
        if (expectedImageOutput[i][indexMax] == 1)
        {
            correct++;
        }
        fread(&byte, 1, 1, labelsTrain);
        expectedImageOutput[i][(int)byte] = 1;
    }

    cout << "Correct: " << correct << " / 10000\n"
         << correct / 100.0 << "%\n";

    //Zadanie 4
    cout << "\nLab 3.7\n";

    NeuralNetwork neuralColor(3);
    neuralColor.addLayer(9);
    neuralColor.addLayer(4);

    neuralColor.activationFunctions.push_back(relu);
    neuralColor.activationFunctions.push_back(NULL);

    vector<vector<float>> inputColors;
    vector<vector<float>> expectedOutputColors;

    fstream colorsFile;
    colorsFile.open("../colors/training_colors.txt", ios::in);

    string line;
    while (getline(colorsFile, line))
    {
        vector<float> data;
        size_t pos = 0;
        string token;
        while ((pos = line.find(" ")) != std::string::npos)
        {
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

    float percentage;

    fstream colorsFileTest;
    colorsFileTest.open("../colors/test_colors.txt", ios::in);

    while (getline(colorsFileTest, line))
    {
        vector<float> data;
        size_t pos = 0;
        string token;
        while ((pos = line.find(" ")) != std::string::npos)
        {
            token = line.substr(0, pos);
            // std::cout << token << std::endl;
            data.push_back(stof(token));
            line.erase(0, pos + 1);
        }
        inputColorsTest.push_back(data);
        expectedOutputColorsTest.push_back(stof(line));
    }

    vector<vector<float>> resultSeries;

    resultSeries = neuralColor.teachForSeries(30, inputColors, 0.01, expectedOutputColors);

    correct = 0;
    for (int i = 0; i < (int)expectedOutputColorsTest.size(); i++)
    {
        vector<float> resultColor = neuralColor.predict(inputColorsTest[i], lo);
        float max = resultColor[0];
        int maxIndex = 0;
        for (int m = 0; m < (int)resultColor.size(); m++)
        {
            if (resultColor[m] > max)
            {
                max = resultColor[m];
                maxIndex = m;
            }
        }

        if (maxIndex + 1 == expectedOutputColorsTest[i])
        {
            correct++;
        }
    }

    cout << "Correct: " << correct;
    cout << " / " << expectedOutputColorsTest.size() << "\n";
    percentage = correct / (float)expectedOutputColorsTest.size();
    cout << percentage * 100 << "%\n";

    return EXIT_SUCCESS;
}