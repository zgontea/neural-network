#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork() {
    currentInputCount = 3;
}

NeuralNetwork::NeuralNetwork(int inputs) {
    currentInputCount = inputs;
}

NeuralNetwork::~NeuralNetwork() {
}

void NeuralNetwork::addLayer(int n) {
    if(n < 1) {
        return;
    }

    srand(time(NULL));

    int currentLayerIndex = getLayersCount();

    layers.resize(currentLayerIndex + 1);
    layers[currentLayerIndex].resize(n);

    for(int i = 0; i < n; i++) {
        
        for(int w = 0; w < currentInputCount; w++) {
            layers[currentLayerIndex][i].push_back((rand() % 10 - 10)/(float)10);
        }
    }

    int inputNumber = 0;
    cout << "Layer " << currentLayerIndex << ":\n";
    for(vector<float> v : layers[currentLayerIndex]) {
        cout << "Weights for input " << inputNumber << ": ";
        for(float value : v) {
            cout << value << ' ';
        }
        cout << '\n';
        inputNumber++;
    }

    cout << '\n';

    currentInputCount = n;
}

vector<float> NeuralNetwork::predict(const vector<float> &input) {
    vector<float> output;
    if(getLayersCount() == 0) {
        return output;
    }

    return deep_neural_network(input, layers);
}

void NeuralNetwork::loadWeights(string fileName) {
    ifstream ifs(fileName);
    if(ifs.fail()) {
        cout << "File not found\n";
        return;
    }
    
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);

    int layersCount = obj["layersCount"].asInt();

    layers.resize(layersCount);

    for(int layerIndex = 0; layerIndex < layersCount; layerIndex++) {
        int neuronsCount = obj["layers"][layerIndex]["neuronsCount"].asInt();
        layers[layerIndex].resize(neuronsCount);

        int inputsCount = obj["layers"][layerIndex]["inputsCount"].asInt();
        for(int y = 0; y < neuronsCount; y++) {
            layers[layerIndex][y].resize(inputsCount);

            for(int x = 0; x < inputsCount; x++) {
                layers[layerIndex][y][x] = obj["layers"][layerIndex]["weights"][(y * inputsCount) + x].asFloat();
            }
        }

        int inputNumber = 0;
        cout << "Layer " << layerIndex << ":\n";
        for(vector<float> v : layers[layerIndex]) {
            cout << "Weights for input " << inputNumber << ": ";
            for(float value : v) {
                cout << value << ' ';
            }
            cout << '\n';
            inputNumber++;
        }

        currentInputCount = neuronsCount;
    }
}

vector<float> NeuralNetwork::teach(int epoch, const vector<float> &input, float learningRatio, const vector<float> &expectedOutput) {
    vector<float> output;
    if(epoch < 1 || input.size() < 1 || learningRatio < 0 || learningRatio > 1 || expectedOutput.size() < 1) {
        return output;
    }

    for(int i = 0; i < epoch; i++) {
        vector<float> prediction = predict(input);
        vector<float> delta;
        delta.resize(prediction.size());

        output = prediction;

        float error = 0;

        for(int d = 0; d < (int)prediction.size(); d++) {
            delta[d] = (2 / (float)prediction.size()) * (prediction[d] - expectedOutput[d]);
            error += (prediction[d] - expectedOutput[d]) * (prediction[d] - expectedOutput[d]);
        }

        error = error / (float)output.size();
        
        cout << "error (epoch " << i << "): " << error << "\n";
        
        vector<vector<float>> outerProduct = outer_product(delta, input);

        for(int y = 0; y < (int)outerProduct.size(); y++) {
            for(int x = 0; x < (int)outerProduct[y].size(); x++) {
                layers[0][y][x] -= (outerProduct[y][x] * learningRatio);
                // cout << layers[0][y][x] << " ";
            }
            // cout << "\n";
        }
    }

    return output;
}

vector<vector<float>> NeuralNetwork::teachForSeries(int epoch, const vector<vector<float>> &input, float learningRatio, const vector<vector<float>> &expectedOutput) {
    vector<vector<float>> output;
    if(epoch < 1 || input.size() < 1 || learningRatio < 0 || learningRatio > 1 || expectedOutput.size() < 1) {
        return output;
    }

    output.resize(input.size());

    for(int i = 0; i < epoch; i++) {
        float totalError = 0;
        for(int s = 0; s < (int)input.size(); s++) {
            float errorForSerie = 0;
            vector<float> prediction = predict(input[s]);
            vector<float> delta;
            delta.resize(prediction.size());

            output[s] = prediction;

            for(int d = 0; d < (int)prediction.size(); d++) {
                delta[d] = (2 / (float)prediction.size()) * (prediction[d] - expectedOutput[s][d]);
                errorForSerie += (prediction[d] - expectedOutput[s][d]) * (prediction[d] - expectedOutput[s][d]);
            }

            // errorForSerie /= (float)prediction.size();
            totalError += errorForSerie;
            vector<vector<float>> outerProduct = outer_product(delta, input[s]);
            // cout << outerProduct.size() << " " << outerProduct[0].size() << "\n";

            for(int y = 0; y < (int)outerProduct.size(); y++) {
                for(int x = 0; x < (int)outerProduct[y].size(); x++) {
                    layers[0][y][x] -= (outerProduct[y][x] * learningRatio);
                    // cout << layers[0][y][x] << " ";
                }
                // cout << "\n";
            }
            // cout << "\n";
        }

        if(i == 999) {
            cout << "error (epoch " << i << "): " << totalError << "\n";
        }
    }

    return output;
}

float relu(const float &value) {
    if(value <= 0) {
        return 0;
    }

    return 1;
}

int NeuralNetwork::getLayersCount() {
    return layers.size(); 
}

int NeuralNetwork::getNeuronsCountByLayer(int layerNumber) {
    if(layerNumber < 0) return -1;

    return layers[layerNumber].size();
}


float NeuralNetwork::neuron(const vector<float> &input, const vector<float> &weights, float bias) {
    if(input.size() == 0 || weights.size() == 0 || input.size() != weights.size()) {
        return -1;
    }

    int inputSize = input.size();
    float result = bias;

    for(int x = 0; x < inputSize; x++) {
        result += activationFunctions(input[x] * weights[x]);
    }

    return result;
}

vector<float> NeuralNetwork::neural_network(const vector<float> &input, const vector<vector<float>> &weights) {
    vector<float> outputs;

    if(weights.size() == 0) {
        return outputs;
    }
    
    if(input.size() != weights[0].size()) {
        return outputs;
    }

    for(vector<float> weight : weights) {
        outputs.push_back(neuron(input, weight, 0));
    }

    return outputs;
}

vector<vector<float>> transpose_matrix(const vector<vector<float>> &matrix) {
    vector<vector<float>> transposed;
    
    if(matrix.size() == 0) {
        return transposed;
    }
    
    transposed.resize(matrix[0].size());

    for(vector<float> &column : transposed) {
      column.resize(matrix.size());
    }
    
    int rows = matrix.size();
    int columns = matrix[0].size();

    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < columns; x++) {
            transposed[x][y] = matrix[y][x];
        }
    }

    return transposed;
}

vector<float> NeuralNetwork::deep_neural_network(const vector<float> &input, const vector<vector<vector<float>>> &weights_for_layers) {
    vector<float> output = input;
    int layers_count = weights_for_layers.size();

    if(layers_count == 0 || input.size() == 0) {
        return output;
    }
    
    for(vector<vector<float>> layer : weights_for_layers) {
        vector<float> new_input = output;
        output = neural_network(new_input, layer);
    }

    return output;
}

vector<vector<float>> outer_product(const vector<float> &v1, const vector<float> &v2) {
    vector<vector<float>> result;
    if(v1.size() < 1 || v2.size() < 1) {
        return result;
    }

    result.resize(v1.size());
    int y = 0;

    for(float value1 : v1) {
        for(float value2 : v2) {
            result[y].push_back(value1 * value2);
        }
        y++;
    }

    return result;
}