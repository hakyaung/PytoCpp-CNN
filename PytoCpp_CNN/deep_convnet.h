#pragma once

#include <vector>
#include <string>
#include "util.h"
#include "optimizer.h"
#include "layers.h"

using namespace std;

using Matrix = vector<vector<float>>;
using Tensor = vector<vector<vector<vector<float>>>>;

class DeepConvNet {
public:
    ParamsDict params;

    Convolution* conv1; Convolution* conv2; Convolution* conv3;
    Convolution* conv4; Convolution* conv5; Convolution* conv6;
    Relu* relu1; Relu* relu2; Relu* relu3; Relu* relu4; Relu* relu5; Relu* relu6; Relu* relu7;
    Pooling* pool1; Pooling* pool2; Pooling* pool3;
    Affine* affine1; Affine* affine2;
    Dropout* drop1; Dropout* drop2;
    SoftmaxLoss* last_layer;

    DeepConvNet();
    ~DeepConvNet();

    Matrix predict(Tensor x, bool train_flg = false);
    float loss(const Tensor& x, const vector<int>& t);
    float accuracy(const Tensor& x, const vector<int>& t, int batch_size = 100);
    ParamsDict gradient(const Tensor& x, const vector<int>& t);

    void save_params(const string& file_name = "params.bin");
    void load_params(const string& file_name = "params.bin");
    void sync_params_to_layers();
};