#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>

using namespace std;

vector<vector<float>> transpose_matrix(const vector<vector<float>> &matrix);
vector<vector<float>> outer_product(const vector<float> &v1, const vector<float> &v2);
float relu(const float &value);

class NeuralNetwork {

    private:
        vector<vector<vector<float>>> layers;
        int currentInputCount;
        
    public:
        NeuralNetwork();
        NeuralNetwork(int inputs);
        ~NeuralNetwork();

        vector<float (*)(const float&)> activationFunctions;

        void addLayer(int n);
        vector<float> predict(const vector<float> &input, const vector<float> &hiddenLayerOutput);
        void loadWeights(string fileName);
        int getLayersCount();
        int getNeuronsCountByLayer(int layerNumber);
        vector<float> teach(int epoch, const vector<float> &input, float learningRatio, const vector<float> &expectedOutput);
        vector<vector<float>> teachForSeries(int epoch, const vector<vector<float>> &input, float learningRatio, const vector<vector<float>> &expectedOutput);
        
        float neuron(const vector<float> &input, const vector<float> &weights, float bias, int layerNum);
        vector<float> neural_network(const vector<float> &input, const vector<vector<float>> &weights, int layerNum);
        vector<float> deep_neural_network(const vector<float> &input, const vector<vector<vector<float>>> &weights_for_layers);
};