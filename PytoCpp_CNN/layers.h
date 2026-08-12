// 파일 위치: PytoCpp_CNN/layers.h
// 파일 명: layers.h

#pragma once

#include <vector>
#include "util.h"

using namespace std;

using Matrix = vector<vector<float>>;
using Tensor = vector<vector<vector<vector<float>>>>;

// 1. Relu
class Relu {
public:
    Tensor mask;
    Relu();
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);
};

// 2. Sigmoid
class Sigmoid {
public:
    Tensor out_val;
    Sigmoid();
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout);
};

// 3. Affine
class Affine {
public:
    Matrix W;
    vector<float> b;
    Matrix x_mat;
    vector<int> original_x_shape;
    Matrix dW;
    vector<float> db;

    Affine(const Matrix& W_init, const vector<float>& b_init);
    Matrix forward(const Tensor& x_in);
    Tensor backward(const Matrix& dout);
};

// 4. SoftmaxLoss
class SoftmaxLoss {
public:
    float loss_val;
    Matrix y;
    vector<int> t;

    SoftmaxLoss();
    float forward(const Matrix& x, const vector<int>& t_in);
    Matrix backward(float dout = 1.0f);
};

// 5. Dropout
class Dropout {
public:
    float dropout_ratio;
    Tensor mask;

    Dropout(float ratio = 0.5f);
    Tensor forward(const Tensor& x, bool train_flg = true);
    Tensor backward(const Tensor& dout);
};

// 6. Convolution
class Convolution {
public:
    Tensor W;
    vector<float> b;
    int stride, pad;
    Tensor x;
    Matrix col, col_W;
    Tensor dW;
    vector<float> db;

    Convolution(const Tensor& W_init, const vector<float>& b_init, int stride = 1, int pad = 0);
    Tensor forward(const Tensor& x_in);
    Tensor backward(const Tensor& dout);
};

// 7. Pooling
class Pooling {
public:
    int pool_h, pool_w, stride, pad;
    Tensor x;
    vector<int> arg_max;

    Pooling(int pool_h, int pool_w, int stride = 2, int pad = 0);
    Tensor forward(const Tensor& x_in);
    Tensor backward(const Tensor& dout);
};