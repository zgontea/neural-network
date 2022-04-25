#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork()
{
    currentInputCount = 3;
}

NeuralNetwork::NeuralNetwork(int inputs)
{
    currentInputCount = inputs;
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::addLayer(int n)
{
    if (n < 1)
    {
        return;
    }

    srand(time(NULL));

    int currentLayerIndex = getLayersCount();

    layers.resize(currentLayerIndex + 1);
    layers[currentLayerIndex].resize(n);

    for (int i = 0; i < n; i++)
    {

        for (int w = 0; w < currentInputCount; w++)
        {
            layers[currentLayerIndex][i].push_back((rand() % 10 - 10) / (float)10);
        }
    }

    int inputNumber = 0;
    // cout << "Layer " << currentLayerIndex << ":\n";
    for (vector<float> v : layers[currentLayerIndex])
    {
        // cout << "Weights for input " << inputNumber << ": ";
        for (float value : v)
        {
            // cout << value << ' ';
        }
        // cout << '\n';
        inputNumber++;
    }

    // cout << '\n';

    currentInputCount = n;
}

vector<float> NeuralNetwork::predict(const vector<float> &input, vector<vector<float>> &layersOutput)
{
    vector<float> output;
    if (getLayersCount() == 0)
    {
        return output;
    }

    return deep_neural_network(input, layers, layersOutput);
}

void NeuralNetwork::loadWeights(string fileName)
{
    ifstream ifs(fileName);
    if (ifs.fail())
    {
        cout << "File not found\n";
        return;
    }

    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);

    int layersCount = obj["layersCount"].asInt();

    layers.resize(layersCount);

    for (int layerIndex = 0; layerIndex < layersCount; layerIndex++)
    {
        int neuronsCount = obj["layers"][layerIndex]["neuronsCount"].asInt();
        layers[layerIndex].resize(neuronsCount);

        int inputsCount = obj["layers"][layerIndex]["inputsCount"].asInt();
        for (int y = 0; y < neuronsCount; y++)
        {
            layers[layerIndex][y].resize(inputsCount);

            for (int x = 0; x < inputsCount; x++)
            {
                layers[layerIndex][y][x] = obj["layers"][layerIndex]["weights"][(y * inputsCount) + x].asFloat();
            }
        }

        int inputNumber = 0;
        // cout << "Layer " << layerIndex << ":\n";
        for (vector<float> v : layers[layerIndex])
        {
            // cout << "Weights for input " << inputNumber << ": ";
            for (float value : v)
            {
                cout << value << ' ';
            }
            cout << '\n';
            inputNumber++;
        }

        currentInputCount = neuronsCount;
    }
}

vector<float> NeuralNetwork::teach(int epoch, const vector<float> &input, float learningRatio, const vector<float> &expectedOutput)
{
    vector<float> output;
    if (epoch < 1 || input.size() < 1 || learningRatio < 0 || learningRatio > 1 || expectedOutput.size() < 1)
    {
        return output;
    }

    for (int i = 0; i < epoch; i++)
    {
        vector<vector<float>> layersOutput;
        vector<float> prediction = predict(input, layersOutput);
        vector<float> delta;
        delta.resize(prediction.size());

        output = prediction;

        float error = 0;

        for (int d = 0; d < (int)prediction.size(); d++)
        {
            delta[d] = (2 / (float)prediction.size()) * (prediction[d] - expectedOutput[d]);
            error += (prediction[d] - expectedOutput[d]) * (prediction[d] - expectedOutput[d]);
        }

        error = error / (float)output.size();

        cout << "error (epoch " << i << "): " << error << "\n";

        vector<vector<float>> outerProduct = outer_product(delta, input);

        for (int y = 0; y < (int)outerProduct.size(); y++)
        {
            for (int x = 0; x < (int)outerProduct[y].size(); x++)
            {
                layers[0][y][x] -= (outerProduct[y][x] * learningRatio);
                // cout << layers[0][y][x] << " ";
            }
            // cout << "\n";
        }
    }

    return output;
}

vector<vector<float>> NeuralNetwork::teachForSeries(int epoch, const vector<vector<float>> &input, float learningRatio, const vector<vector<float>> &expectedOutput)
{
    vector<vector<float>> output;
    if (epoch < 1 || input.size() < 1 || learningRatio < 0 || learningRatio > 1 || expectedOutput.size() < 1)
    {
        return output;
    }

    output.resize(input.size());

    for (int i = 0; i < epoch; i++)
    {
        float totalError = 0;
        for (int s = 0; s < (int)input.size(); s++)
        {
            float errorForSerie = 0;
            vector<vector<float>> layersOutput;
            vector<float> prediction = predict(input[s], layersOutput);
            vector<float> delta;
            delta.resize(prediction.size());

            output[s] = prediction;

            for (int d = 0; d < (int)prediction.size(); d++)
            {
                delta[d] = (2 / (float)prediction.size()) * (prediction[d] - expectedOutput[s][d]);
                errorForSerie += (prediction[d] - expectedOutput[s][d]) * (prediction[d] - expectedOutput[s][d]);
            }

            for (int l = getLayersCount() - 2; l >= 0; l--)
            {
                vector<vector<float>> transposedWeights = transpose_matrix(layers[l + 1]);
                vector<float> currLayerDelta = multiply_matrix(transposedWeights, delta);
                vector<float> layerOutputDeriv;
                layerOutputDeriv.resize(layersOutput[l].size());
                for (int i = 0; i < layerOutputDeriv.size(); i++)
                {
                    layerOutputDeriv[i] = reluDeriv(layersOutput[l][i]);
                }
                currLayerDelta = multiply_matrix_deriv(currLayerDelta, layerOutputDeriv);
                vector<float> layerInput;
                if (l == 0)
                {
                    layerInput = input[s];
                }
                else
                {
                    layerInput = layersOutput[l - 1];
                }
                vector<vector<float>> outerProduct = outer_product(currLayerDelta, layerInput);
                for (int y = 0; y < (int)outerProduct.size(); y++)
                {
                    for (int x = 0; x < (int)outerProduct[y].size(); x++)
                    {
                        layers[l][y][x] -= (outerProduct[y][x] * learningRatio);
                        // cout << layers[0][y][x] << " ";
                    }
                    // cout << "\n";
                }
            }

            // errorForSerie /= (float)prediction.size();
            totalError += errorForSerie;
            vector<vector<float>> outerProduct = outer_product(delta, layersOutput[layers.size() - 2]);
            // cout << outerProduct.size() << " " << outerProduct[0].size() << "\n";

            for (int y = 0; y < (int)outerProduct.size(); y++)
            {
                for (int x = 0; x < (int)outerProduct[y].size(); x++)
                {
                    layers[layers.size() - 1][y][x] -= (outerProduct[y][x] * learningRatio);
                    // cout << layers[0][y][x] << " ";
                }
                // cout << "\n";
            }
            // cout << "\n";
        }
    }

    return output;
}

void NeuralNetwork::fit(const vector<float> &input, const vector<float> &expectedOutput, float learningRatio)
{
    vector<vector<float>> layersOutput;
    vector<float> prediction = predict(input, layersOutput);
    vector<float> delta;
    delta.resize(prediction.size());

    for (int d = 0; d < (int)prediction.size(); d++)
    {
        delta[d] = (2 / (float)prediction.size()) * (prediction[d] - expectedOutput[d]);
    }

    for (int l = getLayersCount() - 2; l >= 0; l--)
    {
        vector<vector<float>> transposedWeights = transpose_matrix(layers[l + 1]);
        vector<float> currLayerDelta = multiply_matrix(transposedWeights, delta);
        vector<float> layerOutputDeriv;
        layerOutputDeriv.resize(layersOutput[l].size());
        for (int i = 0; i < layerOutputDeriv.size(); i++)
        {
            layerOutputDeriv[i] = reluDeriv(layersOutput[l][i]);
        }
        currLayerDelta = multiply_matrix_deriv(currLayerDelta, layerOutputDeriv);
        vector<float> layerInput;
        if (l == 0)
        {
            layerInput = input;
        }
        else
        {
            layerInput = layersOutput[l - 1];
        }
        vector<vector<float>> outerProduct = outer_product(currLayerDelta, layerInput);
        for (int y = 0; y < (int)outerProduct.size(); y++)
        {
            for (int x = 0; x < (int)outerProduct[y].size(); x++)
            {
                layers[l][y][x] -= (outerProduct[y][x] * learningRatio);
                // cout << layers[0][y][x] << " ";
            }
            // cout << "\n";
        }
    }

    // errorForSerie /= (float)prediction.size();
    vector<vector<float>> outerProduct = outer_product(delta, layersOutput[layers.size() - 2]);
    // cout << outerProduct.size() << " " << outerProduct[0].size() << "\n";

    for (int y = 0; y < (int)outerProduct.size(); y++)
    {
        for (int x = 0; x < (int)outerProduct[y].size(); x++)
        {
            layers[layers.size() - 1][y][x] -= (outerProduct[y][x] * learningRatio);
            // cout << layers[0][y][x] << " ";
        }
        // cout << "\n";
    }
}

float relu(const float &value)
{
    if (value <= 0)
    {
        return 0;
    }

    return value;
}

float reluDeriv(const float &value)
{
    if (value <= 0)
    {
        return 0;
    }

    return 1;
}

int NeuralNetwork::getLayersCount()
{
    return layers.size();
}

int NeuralNetwork::getNeuronsCountByLayer(int layerNumber)
{
    if (layerNumber < 0)
        return -1;

    return layers[layerNumber].size();
}

float NeuralNetwork::neuron(const vector<float> &input, const vector<float> &weights, float bias, int layerNum)
{
    if (input.size() == 0 || weights.size() == 0 || input.size() != weights.size())
    {
        return -1;
    }

    int inputSize = input.size();
    float result = bias;

    for (int x = 0; x < inputSize; x++)
    {
        result += (input[x] * weights[x]);
    }

    if (activationFunctions[layerNum])
    {
        result = activationFunctions[layerNum](result);
    }

    return result;
}

vector<float> NeuralNetwork::neural_network(const vector<float> &input, const vector<vector<float>> &weights, int layerNum)
{
    vector<float> outputs;

    if (weights.size() == 0)
    {
        return outputs;
    }

    if (input.size() != weights[0].size())
    {
        return outputs;
    }

    for (vector<float> weight : weights)
    {
        outputs.push_back(neuron(input, weight, 0, layerNum));
    }

    return outputs;
}

vector<vector<float>> transpose_matrix(const vector<vector<float>> &matrix)
{
    vector<vector<float>> transposed;

    if (matrix.size() == 0)
    {
        return transposed;
    }

    transposed.resize(matrix[0].size());

    for (vector<float> &column : transposed)
    {
        column.resize(matrix.size());
    }

    int rows = matrix.size();
    int columns = matrix[0].size();

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < columns; x++)
        {
            transposed[x][y] = matrix[y][x];
        }
    }

    return transposed;
}

vector<float> NeuralNetwork::deep_neural_network(const vector<float> &input, const vector<vector<vector<float>>> &weights_for_layers, vector<vector<float>> &layersOutput)
{
    vector<float> output = input;
    int layers_count = weights_for_layers.size();

    if (layers_count == 0 || input.size() == 0)
    {
        return output;
    }

    int layerNum = 0;

    for (vector<vector<float>> layer : weights_for_layers)
    {
        vector<float> new_input = output;
        output = neural_network(new_input, layer, layerNum);
        layersOutput.push_back(output);
        layerNum++;
    }

    return output;
}

vector<vector<float>> outer_product(const vector<float> &v1, const vector<float> &v2)
{
    vector<vector<float>> result;
    if (v1.size() < 1 || v2.size() < 1)
    {
        return result;
    }

    result.resize(v1.size());
    int y = 0;

    for (float value1 : v1)
    {
        for (float value2 : v2)
        {
            result[y].push_back(value1 * value2);
        }
        y++;
    }

    return result;
}

vector<float> multiply_matrix(const vector<vector<float>> &weights, const vector<float> &delta)
{
    vector<float> output;
    output.resize(weights.size());

    for (int y = 0; y < weights.size(); y++)
    {
        for (int x = 0; x < weights[y].size(); x++)
        {
            output[y] += weights[y][x] * delta[x];
        }
    }

    return output;
}

vector<float> multiply_matrix_deriv(const vector<float> &v1, const vector<float> &v2)
{
    vector<float> output;
    output.resize(v1.size());

    for (int y = 0; y < v1.size(); y++)
    {
        output[y] = v1[y] * v2[y];
    }

    return output;
}