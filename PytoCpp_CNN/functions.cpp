#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

using Matrix = vector<vector<float>>;

// 항등 함수
Matrix identity_function(const Matrix& x) {
    return x;
}

// 계단 함수
Matrix step_function(const Matrix& x) {
    int N = x.size(), D = x[0].size();
    Matrix out(N, vector<float>(D, 0.0f));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            out[i][j] = (x[i][j] > 0.0f) ? 1.0f : 0.0f;
        }
    }
    return out;
}

// 시그모이드
Matrix sigmoid(const Matrix& x) {
    int N = x.size(), D = x[0].size();
    Matrix out(N, vector<float>(D, 0.0f));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            out[i][j] = 1.0f / (1.0f + exp(-x[i][j]));
        }
    }
    return out;
}

// 시그모이드 기울기
Matrix sigmoid_grad(const Matrix& x) {
    Matrix s = sigmoid(x);
    int N = s.size(), D = s[0].size();
    Matrix grad(N, vector<float>(D, 0.0f));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            grad[i][j] = (1.0f - s[i][j]) * s[i][j];
        }
    }
    return grad;
}

// ReLU
Matrix relu(const Matrix& x) {
    int N = x.size(), D = x[0].size();
    Matrix out(N, vector<float>(D, 0.0f));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            out[i][j] = max(0.0f, x[i][j]);
        }
    }
    return out;
}

// ReLU 기울기
Matrix relu_grad(const Matrix& x) {
    int N = x.size(), D = x[0].size();
    Matrix grad(N, vector<float>(D, 0.0f));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            grad[i][j] = (x[i][j] >= 0.0f) ? 1.0f : 0.0f;
        }
    }
    return grad;
}

// 소프트맥스
Matrix softmax(const Matrix& x) {
    int N = x.size(), C = x[0].size();
    Matrix y(N, vector<float>(C, 0.0f));

    for (int i = 0; i < N; ++i) {
        // 오버플로 대책 (행별 최댓값 찾기)
        float max_val = x[i][0];
        for (int j = 1; j < C; ++j) {
            if (x[i][j] > max_val) {
                max_val = x[i][j];
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < C; ++j) {
            y[i][j] = exp(x[i][j] - max_val);
            sum_exp += y[i][j];
        }
        for (int j = 0; j < C; ++j) {
            y[i][j] /= sum_exp;
        }
    }
    return y;
}

// 평균 제곱 오차
float mean_squared_error(const Matrix& y, const Matrix& t) {
    int N = y.size(), C = y[0].size();
    float sum = 0.0f;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
            float diff = y[i][j] - t[i][j];
            sum += diff * diff;
        }
    }
    return 0.5f * sum;
}

// 교차 엔트로피
// 파이썬 원본은 원-핫과 단일 라벨을 모두 처리하지만,
// 앞서 작성하신 SoftmaxLoss 구조에 맞춰 t를 1차원 라벨 인덱스로 받게 구현
float cross_entropy_error(const Matrix& y, const vector<int>& t) {
    int batch_size = y.size();
    float sum = 0.0f;

    for (int i = 0; i < batch_size; ++i) {
        // y[np.arange(batch_size), t] 에 해당하는 2차원 인덱싱
        sum -= log(y[i][t[i]] + 1e-7f);
    }
    return sum / batch_size;
}

// softmax loss 
float softmax_loss(const Matrix& X, const vector<int>& t) {
    Matrix y = softmax(X);
    return cross_entropy_error(y, t);
}