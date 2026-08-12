// 파일 위치: PytoCpp_CNN/optimizer.cpp
// 파일 명: optimizer.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include "util.h"
#include "optimizer.h"

using namespace std;

// ==========================================
// 경사하강법 SGD
// ==========================================
SGD::SGD(float lr) {
    this->lr = lr;
}

void SGD::update(ParamsDict& params, ParamsDict& grads) {
    // 1. Convolution (4D)
    for (auto& pair : params.conv_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.conv_W[key].size(); ++i)
            for (size_t j = 0; j < params.conv_W[key][0].size(); ++j)
                for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k)
                    for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l)
                        params.conv_W[key][i][j][k][l] -= lr * grads.conv_W[key][i][j][k][l];
    }
    // 2. Affine (2D)
    for (auto& pair : params.affine_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.affine_W[key].size(); ++i)
            for (size_t j = 0; j < params.affine_W[key][0].size(); ++j)
                params.affine_W[key][i][j] -= lr * grads.affine_W[key][i][j];
    }
    // 3. Bias (1D)
    for (auto& pair : params.b) {
        string key = pair.first;
        for (size_t i = 0; i < params.b[key].size(); ++i)
            params.b[key][i] -= lr * grads.b[key][i];
    }
}

// ==========================================
// 모멘텀 Momentum
// ==========================================
Momentum::Momentum(float lr, float momentum) {
    this->lr = lr;
    this->momentum = momentum;
}

void Momentum::update(ParamsDict& params, ParamsDict& grads) {
    if (v.empty()) {
        for (auto& pair : params.conv_W) {
            auto& p = pair.second;
            v.conv_W[pair.first] = Tensor(p.size(), vector<vector<vector<float>>>(p[0].size(), vector<vector<float>>(p[0][0].size(), vector<float>(p[0][0][0].size(), 0.0f))));
        }
        for (auto& pair : params.affine_W) {
            auto& p = pair.second;
            v.affine_W[pair.first] = Matrix(p.size(), vector<float>(p[0].size(), 0.0f));
        }
        for (auto& pair : params.b) {
            v.b[pair.first] = vector<float>(pair.second.size(), 0.0f);
        }
    }

    // 1. Convolution (4D)
    for (auto& pair : params.conv_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.conv_W[key].size(); ++i)
            for (size_t j = 0; j < params.conv_W[key][0].size(); ++j)
                for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k)
                    for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l) {
                        v.conv_W[key][i][j][k][l] = momentum * v.conv_W[key][i][j][k][l] - lr * grads.conv_W[key][i][j][k][l];
                        params.conv_W[key][i][j][k][l] += v.conv_W[key][i][j][k][l];
                    }
    }
    // 2. Affine (2D)
    for (auto& pair : params.affine_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.affine_W[key].size(); ++i)
            for (size_t j = 0; j < params.affine_W[key][0].size(); ++j) {
                v.affine_W[key][i][j] = momentum * v.affine_W[key][i][j] - lr * grads.affine_W[key][i][j];
                params.affine_W[key][i][j] += v.affine_W[key][i][j];
            }
    }
    // 3. Bias (1D)
    for (auto& pair : params.b) {
        string key = pair.first;
        for (size_t i = 0; i < params.b[key].size(); ++i) {
            v.b[key][i] = momentum * v.b[key][i] - lr * grads.b[key][i];
            params.b[key][i] += v.b[key][i];
        }
    }
}

// ==========================================
// 네스테로프 Nesterov
// ==========================================
Nesterov::Nesterov(float lr, float momentum) {
    this->lr = lr;
    this->momentum = momentum;
}

void Nesterov::update(ParamsDict& params, ParamsDict& grads) {
    if (v.empty()) {
        for (auto& pair : params.conv_W) {
            auto& p = pair.second;
            v.conv_W[pair.first] = Tensor(p.size(), vector<vector<vector<float>>>(p[0].size(), vector<vector<float>>(p[0][0].size(), vector<float>(p[0][0][0].size(), 0.0f))));
        }
        for (auto& pair : params.affine_W) {
            auto& p = pair.second;
            v.affine_W[pair.first] = Matrix(p.size(), vector<float>(p[0].size(), 0.0f));
        }
        for (auto& pair : params.b) {
            v.b[pair.first] = vector<float>(pair.second.size(), 0.0f);
        }
    }

    // 1. Convolution (4D)
    for (auto& pair : params.conv_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.conv_W[key].size(); ++i)
            for (size_t j = 0; j < params.conv_W[key][0].size(); ++j)
                for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k)
                    for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l) {
                        v.conv_W[key][i][j][k][l] *= momentum;
                        v.conv_W[key][i][j][k][l] -= lr * grads.conv_W[key][i][j][k][l];
                        params.conv_W[key][i][j][k][l] += momentum * momentum * v.conv_W[key][i][j][k][l];
                        params.conv_W[key][i][j][k][l] -= (1.0f + momentum) * lr * grads.conv_W[key][i][j][k][l];
                    }
    }
    // 2. Affine (2D)
    for (auto& pair : params.affine_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.affine_W[key].size(); ++i)
            for (size_t j = 0; j < params.affine_W[key][0].size(); ++j) {
                v.affine_W[key][i][j] *= momentum;
                v.affine_W[key][i][j] -= lr * grads.affine_W[key][i][j];
                params.affine_W[key][i][j] += momentum * momentum * v.affine_W[key][i][j];
                params.affine_W[key][i][j] -= (1.0f + momentum) * lr * grads.affine_W[key][i][j];
            }
    }
    // 3. Bias (1D)
    for (auto& pair : params.b) {
        string key = pair.first;
        for (size_t i = 0; i < params.b[key].size(); ++i) {
            v.b[key][i] *= momentum;
            v.b[key][i] -= lr * grads.b[key][i];
            params.b[key][i] += momentum * momentum * v.b[key][i];
            params.b[key][i] -= (1.0f + momentum) * lr * grads.b[key][i];
        }
    }
}

// ==========================================
// 에이다그레드 AdaGrad
// ==========================================
AdaGrad::AdaGrad(float lr) {
    this->lr = lr;
}

void AdaGrad::update(ParamsDict& params, ParamsDict& grads) {
    if (h.empty()) {
        for (auto& pair : params.conv_W) {
            auto& p = pair.second;
            h.conv_W[pair.first] = Tensor(p.size(), vector<vector<vector<float>>>(p[0].size(), vector<vector<float>>(p[0][0].size(), vector<float>(p[0][0][0].size(), 0.0f))));
        }
        for (auto& pair : params.affine_W) {
            auto& p = pair.second;
            h.affine_W[pair.first] = Matrix(p.size(), vector<float>(p[0].size(), 0.0f));
        }
        for (auto& pair : params.b) {
            h.b[pair.first] = vector<float>(pair.second.size(), 0.0f);
        }
    }

    // 1. Convolution (4D)
    for (auto& pair : params.conv_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.conv_W[key].size(); ++i)
            for (size_t j = 0; j < params.conv_W[key][0].size(); ++j)
                for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k)
                    for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l) {
                        h.conv_W[key][i][j][k][l] += grads.conv_W[key][i][j][k][l] * grads.conv_W[key][i][j][k][l];
                        params.conv_W[key][i][j][k][l] -= lr * grads.conv_W[key][i][j][k][l] / (sqrt(h.conv_W[key][i][j][k][l]) + 1e-7f);
                    }
    }
    // 2. Affine (2D)
    for (auto& pair : params.affine_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.affine_W[key].size(); ++i)
            for (size_t j = 0; j < params.affine_W[key][0].size(); ++j) {
                h.affine_W[key][i][j] += grads.affine_W[key][i][j] * grads.affine_W[key][i][j];
                params.affine_W[key][i][j] -= lr * grads.affine_W[key][i][j] / (sqrt(h.affine_W[key][i][j]) + 1e-7f);
            }
    }
    // 3. Bias (1D)
    for (auto& pair : params.b) {
        string key = pair.first;
        for (size_t i = 0; i < params.b[key].size(); ++i) {
            h.b[key][i] += grads.b[key][i] * grads.b[key][i];
            params.b[key][i] -= lr * grads.b[key][i] / (sqrt(h.b[key][i]) + 1e-7f);
        }
    }
}

// ==========================================
// RMSprop
// ==========================================
RMSprop::RMSprop(float lr, float decay_rate) {
    this->lr = lr;
    this->decay_rate = decay_rate;
}

void RMSprop::update(ParamsDict& params, ParamsDict& grads) {
    if (h.empty()) {
        for (auto& pair : params.conv_W) {
            auto& p = pair.second;
            h.conv_W[pair.first] = Tensor(p.size(), vector<vector<vector<float>>>(p[0].size(), vector<vector<float>>(p[0][0].size(), vector<float>(p[0][0][0].size(), 0.0f))));
        }
        for (auto& pair : params.affine_W) {
            auto& p = pair.second;
            h.affine_W[pair.first] = Matrix(p.size(), vector<float>(p[0].size(), 0.0f));
        }
        for (auto& pair : params.b) {
            h.b[pair.first] = vector<float>(pair.second.size(), 0.0f);
        }
    }

    // 1. Convolution (4D)
    for (auto& pair : params.conv_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.conv_W[key].size(); ++i)
            for (size_t j = 0; j < params.conv_W[key][0].size(); ++j)
                for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k)
                    for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l) {
                        h.conv_W[key][i][j][k][l] *= decay_rate;
                        h.conv_W[key][i][j][k][l] += (1.0f - decay_rate) * grads.conv_W[key][i][j][k][l] * grads.conv_W[key][i][j][k][l];
                        params.conv_W[key][i][j][k][l] -= lr * grads.conv_W[key][i][j][k][l] / (sqrt(h.conv_W[key][i][j][k][l]) + 1e-7f);
                    }
    }
    // 2. Affine (2D)
    for (auto& pair : params.affine_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.affine_W[key].size(); ++i)
            for (size_t j = 0; j < params.affine_W[key][0].size(); ++j) {
                h.affine_W[key][i][j] *= decay_rate;
                h.affine_W[key][i][j] += (1.0f - decay_rate) * grads.affine_W[key][i][j] * grads.affine_W[key][i][j];
                params.affine_W[key][i][j] -= lr * grads.affine_W[key][i][j] / (sqrt(h.affine_W[key][i][j]) + 1e-7f);
            }
    }
    // 3. Bias (1D)
    for (auto& pair : params.b) {
        string key = pair.first;
        for (size_t i = 0; i < params.b[key].size(); ++i) {
            h.b[key][i] *= decay_rate;
            h.b[key][i] += (1.0f - decay_rate) * grads.b[key][i] * grads.b[key][i];
            params.b[key][i] -= lr * grads.b[key][i] / (sqrt(h.b[key][i]) + 1e-7f);
        }
    }
}

// ==========================================
// Adam
// ==========================================
Adam::Adam(float lr, float beta1, float beta2) {
    this->lr = lr;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->iter = 0;
}

void Adam::update(ParamsDict& params, ParamsDict& grads) {
    if (m.empty()) {
        for (auto& pair : params.conv_W) {
            auto& p = pair.second;
            m.conv_W[pair.first] = Tensor(p.size(), vector<vector<vector<float>>>(p[0].size(), vector<vector<float>>(p[0][0].size(), vector<float>(p[0][0][0].size(), 0.0f))));
            v.conv_W[pair.first] = m.conv_W[pair.first];
        }
        for (auto& pair : params.affine_W) {
            auto& p = pair.second;
            m.affine_W[pair.first] = Matrix(p.size(), vector<float>(p[0].size(), 0.0f));
            v.affine_W[pair.first] = m.affine_W[pair.first];
        }
        for (auto& pair : params.b) {
            m.b[pair.first] = vector<float>(pair.second.size(), 0.0f);
            v.b[pair.first] = m.b[pair.first];
        }
    }

    iter++;
    float lr_t = lr * sqrt(1.0f - pow(beta2, iter)) / (1.0f - pow(beta1, iter));

    // 1. Convolution (4D)
    for (auto& pair : params.conv_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.conv_W[key].size(); ++i)
            for (size_t j = 0; j < params.conv_W[key][0].size(); ++j)
                for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k)
                    for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l) {
                        m.conv_W[key][i][j][k][l] += (1.0f - beta1) * (grads.conv_W[key][i][j][k][l] - m.conv_W[key][i][j][k][l]);
                        v.conv_W[key][i][j][k][l] += (1.0f - beta2) * (grads.conv_W[key][i][j][k][l] * grads.conv_W[key][i][j][k][l] - v.conv_W[key][i][j][k][l]);
                        params.conv_W[key][i][j][k][l] -= lr_t * m.conv_W[key][i][j][k][l] / (sqrt(v.conv_W[key][i][j][k][l]) + 1e-7f);
                    }
    }
    // 2. Affine (2D)
    for (auto& pair : params.affine_W) {
        string key = pair.first;
        for (size_t i = 0; i < params.affine_W[key].size(); ++i)
            for (size_t j = 0; j < params.affine_W[key][0].size(); ++j) {
                m.affine_W[key][i][j] += (1.0f - beta1) * (grads.affine_W[key][i][j] - m.affine_W[key][i][j]);
                v.affine_W[key][i][j] += (1.0f - beta2) * (grads.affine_W[key][i][j] * grads.affine_W[key][i][j] - v.affine_W[key][i][j]);
                params.affine_W[key][i][j] -= lr_t * m.affine_W[key][i][j] / (sqrt(v.affine_W[key][i][j]) + 1e-7f);
            }
    }
    // 3. Bias (1D)
    for (auto& pair : params.b) {
        string key = pair.first;
        for (size_t i = 0; i < params.b[key].size(); ++i) {
            m.b[key][i] += (1.0f - beta1) * (grads.b[key][i] - m.b[key][i]);
            v.b[key][i] += (1.0f - beta2) * (grads.b[key][i] * grads.b[key][i] - v.b[key][i]);
            params.b[key][i] -= lr_t * m.b[key][i] / (sqrt(v.b[key][i]) + 1e-7f);
        }
    }
}