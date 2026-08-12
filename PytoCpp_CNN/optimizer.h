// 파일 위치: PytoCpp_CNN/optimizer.h
// 파일 명: optimizer.h

#pragma once

#include <vector>
#include <string>
#include <map>
#include "util.h"

using namespace std;

using Matrix = vector<vector<float>>;
using Tensor = vector<vector<vector<vector<float>>>>;

// DeepConvNet과 옵티마이저가 주고받을 파이썬 dict의 대체재
struct ParamsDict {
    map<string, Tensor> conv_W;     // 4차원 가중치 (Convolution)
    map<string, Matrix> affine_W;   // 2차원 가중치 (Affine)
    map<string, vector<float>> b;   // 1차원 편향

    bool empty() const {
        return conv_W.empty() && affine_W.empty() && b.empty();
    }
};

// 1. SGD
class SGD {
public:
    float lr;
    SGD(float lr = 0.01f);
    void update(ParamsDict& params, ParamsDict& grads);
};

// 2. Momentum
class Momentum {
public:
    float lr;
    float momentum;
    ParamsDict v;
    Momentum(float lr = 0.01f, float momentum = 0.9f);
    void update(ParamsDict& params, ParamsDict& grads);
};

// 3. Nesterov
class Nesterov {
public:
    float lr;
    float momentum;
    ParamsDict v;
    Nesterov(float lr = 0.01f, float momentum = 0.9f);
    void update(ParamsDict& params, ParamsDict& grads);
};

// 4. AdaGrad
class AdaGrad {
public:
    float lr;
    ParamsDict h;
    AdaGrad(float lr = 0.01f);
    void update(ParamsDict& params, ParamsDict& grads);
};

// 5. RMSprop
class RMSprop {
public:
    float lr;
    float decay_rate;
    ParamsDict h;
    RMSprop(float lr = 0.01f, float decay_rate = 0.99f);
    void update(ParamsDict& params, ParamsDict& grads);
};

// 6. Adam
class Adam {
public:
    float lr;
    float beta1;
    float beta2;
    int iter;
    ParamsDict m;
    ParamsDict v;
    Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f);
    void update(ParamsDict& params, ParamsDict& grads);
};