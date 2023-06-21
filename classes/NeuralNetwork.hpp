#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

typedef float Scalar;
typedef arma::mat Matrix;
typedef arma::rowvec RowVec;
typedef arma::colvec ColVec;

class NeuralNetwork {
public:
    NeuralNetwork(vector<uint> topology, Scalar learningRate = Scalar(0.005));
    void Forward(RowVec& input);
    void Backward(RowVec& input);
    void calcDeltas(RowVec& output);
    void updateWs();
    void train(vector<RowVec*> data);

    vector<RowVec> neuronLayers;
    vector<RowVec> cacheLayers;
    vector<RowVec> deltas;
    vector<RowVec> weights;
};
