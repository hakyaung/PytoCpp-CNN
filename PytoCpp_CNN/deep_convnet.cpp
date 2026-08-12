#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include "deep_convnet.h"

using namespace std;

// 헬퍼 함수들 (cpp 내부에서만 사용되도록 static으로 분리)
static Tensor mat2tensor(const Matrix& m) {
    int N = m.size(), D = m[0].size();
    Tensor t(N, vector<vector<vector<float>>>(D, vector<vector<float>>(1, vector<float>(1))));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            t[i][j][0][0] = m[i][j];
    return t;
}

static Matrix tensor2mat(const Tensor& t) {
    int N = t.size(), D = t[0].size();
    Matrix m(N, vector<float>(D));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            m[i][j] = t[i][j][0][0];
    return m;
}

static Tensor randn_tensor(int FN, int C, int FH, int FW, float scale) {
    Tensor t(FN, vector<vector<vector<float>>>(C, vector<vector<float>>(FH, vector<float>(FW))));
    random_device rd; mt19937 gen(rd()); normal_distribution<float> dist(0.0f, 1.0f);
    for (int fn = 0; fn < FN; fn++)
        for (int c = 0; c < C; c++)
            for (int fh = 0; fh < FH; fh++)
                for (int fw = 0; fw < FW; fw++)
                    t[fn][c][fh][fw] = dist(gen) * scale;
    return t;
}

static Matrix randn_matrix(int row, int col, float scale) {
    Matrix m(row, vector<float>(col));
    random_device rd; mt19937 gen(rd()); normal_distribution<float> dist(0.0f, 1.0f);
    for (int r = 0; r < row; r++)
        for (int c = 0; c < col; c++)
            m[r][c] = dist(gen) * scale;
    return m;
}

// 생성자
DeepConvNet::DeepConvNet() {
    int hidden_size = 50;
    int output_size = 10;
    vector<int> pre_node_nums = { 1 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 4 * 4, hidden_size };
    vector<float> weight_scales;
    for (int n : pre_node_nums) weight_scales.push_back(sqrt(2.0f / n));

    params.conv_W["W1"] = randn_tensor(16, 1, 3, 3, weight_scales[0]);
    params.b["b1"] = vector<float>(16, 0.0f);
    params.conv_W["W2"] = randn_tensor(16, 16, 3, 3, weight_scales[1]);
    params.b["b2"] = vector<float>(16, 0.0f);
    params.conv_W["W3"] = randn_tensor(32, 16, 3, 3, weight_scales[2]);
    params.b["b3"] = vector<float>(32, 0.0f);
    params.conv_W["W4"] = randn_tensor(32, 32, 3, 3, weight_scales[3]);
    params.b["b4"] = vector<float>(32, 0.0f);
    params.conv_W["W5"] = randn_tensor(64, 32, 3, 3, weight_scales[4]);
    params.b["b5"] = vector<float>(64, 0.0f);
    params.conv_W["W6"] = randn_tensor(64, 64, 3, 3, weight_scales[5]);
    params.b["b6"] = vector<float>(64, 0.0f);
    params.affine_W["W7"] = randn_matrix(64 * 4 * 4, hidden_size, weight_scales[6]);
    params.b["b7"] = vector<float>(hidden_size, 0.0f);
    params.affine_W["W8"] = randn_matrix(hidden_size, output_size, weight_scales[7]);
    params.b["b8"] = vector<float>(output_size, 0.0f);

    conv1 = new Convolution(params.conv_W["W1"], params.b["b1"], 1, 1); relu1 = new Relu();
    conv2 = new Convolution(params.conv_W["W2"], params.b["b2"], 1, 1); relu2 = new Relu();
    pool1 = new Pooling(2, 2, 2, 0);
    conv3 = new Convolution(params.conv_W["W3"], params.b["b3"], 1, 1); relu3 = new Relu();
    conv4 = new Convolution(params.conv_W["W4"], params.b["b4"], 1, 2); relu4 = new Relu();
    pool2 = new Pooling(2, 2, 2, 0);
    conv5 = new Convolution(params.conv_W["W5"], params.b["b5"], 1, 1); relu5 = new Relu();
    conv6 = new Convolution(params.conv_W["W6"], params.b["b6"], 1, 1); relu6 = new Relu();
    pool3 = new Pooling(2, 2, 2, 0);
    affine1 = new Affine(params.affine_W["W7"], params.b["b7"]); relu7 = new Relu(); drop1 = new Dropout(0.5f);
    affine2 = new Affine(params.affine_W["W8"], params.b["b8"]); drop2 = new Dropout(0.5f);
    last_layer = new SoftmaxLoss();
}

// 소멸자
DeepConvNet::~DeepConvNet() {
    delete conv1; delete relu1; delete conv2; delete relu2; delete pool1;
    delete conv3; delete relu3; delete conv4; delete relu4; delete pool2;
    delete conv5; delete relu5; delete conv6; delete relu6; delete pool3;
    delete affine1; delete relu7; delete drop1;
    delete affine2; delete drop2; delete last_layer;
}

Matrix DeepConvNet::predict(Tensor x, bool train_flg) {
    x = conv1->forward(x);  x = relu1->forward(x);
    x = conv2->forward(x);  x = relu2->forward(x);
    x = pool1->forward(x);
    x = conv3->forward(x);  x = relu3->forward(x);
    x = conv4->forward(x);  x = relu4->forward(x);
    x = pool2->forward(x);
    x = conv5->forward(x);  x = relu5->forward(x);
    x = conv6->forward(x);  x = relu6->forward(x);
    x = pool3->forward(x);

    Matrix a1 = affine1->forward(x);
    Tensor a1_t = mat2tensor(a1);
    a1_t = relu7->forward(a1_t);
    a1_t = drop1->forward(a1_t, train_flg);

    Matrix a2 = affine2->forward(a1_t);
    Tensor a2_t = mat2tensor(a2);
    a2_t = drop2->forward(a2_t, train_flg);

    return tensor2mat(a2_t);
}

float DeepConvNet::loss(const Tensor& x, const vector<int>& t) {
    Matrix y = predict(x, true);
    return last_layer->forward(y, t);
}

float DeepConvNet::accuracy(const Tensor& x, const vector<int>& t, int batch_size) {
    float acc = 0.0f;
    int N = x.size();
    for (int i = 0; i < N / batch_size; i++) {
        Tensor tx(x.begin() + i * batch_size, x.begin() + (i + 1) * batch_size);
        vector<int> tt(t.begin() + i * batch_size, t.begin() + (i + 1) * batch_size);
        Matrix y = predict(tx, false);
        for (int j = 0; j < batch_size; j++) {
            int max_idx = 0;
            float max_val = y[j][0];
            for (size_t c = 1; c < y[j].size(); c++) {
                if (y[j][c] > max_val) {
                    max_val = y[j][c];
                    max_idx = c;
                }
            }
            if (max_idx == tt[j]) acc += 1.0f;
        }
    }
    return acc / N;
}

ParamsDict DeepConvNet::gradient(const Tensor& x, const vector<int>& t) {
    loss(x, t);
    Matrix dout_mat = last_layer->backward(1.0f);
    Tensor dout = mat2tensor(dout_mat);

    dout = drop2->backward(dout);
    dout_mat = tensor2mat(dout);
    dout = affine2->backward(dout_mat);

    dout = drop1->backward(dout);
    dout = relu7->backward(dout);
    dout_mat = tensor2mat(dout);
    dout = affine1->backward(dout_mat);

    dout = pool3->backward(dout); dout = relu6->backward(dout); dout = conv6->backward(dout);
    dout = relu5->backward(dout); dout = conv5->backward(dout);
    dout = pool2->backward(dout); dout = relu4->backward(dout); dout = conv4->backward(dout);
    dout = relu3->backward(dout); dout = conv3->backward(dout);
    dout = pool1->backward(dout); dout = relu2->backward(dout); dout = conv2->backward(dout);
    dout = relu1->backward(dout); dout = conv1->backward(dout);

    ParamsDict grads;
    grads.conv_W["W1"] = conv1->dW; grads.b["b1"] = conv1->db;
    grads.conv_W["W2"] = conv2->dW; grads.b["b2"] = conv2->db;
    grads.conv_W["W3"] = conv3->dW; grads.b["b3"] = conv3->db;
    grads.conv_W["W4"] = conv4->dW; grads.b["b4"] = conv4->db;
    grads.conv_W["W5"] = conv5->dW; grads.b["b5"] = conv5->db;
    grads.conv_W["W6"] = conv6->dW; grads.b["b6"] = conv6->db;
    grads.affine_W["W7"] = affine1->dW; grads.b["b7"] = affine1->db;
    grads.affine_W["W8"] = affine2->dW; grads.b["b8"] = affine2->db;

    return grads;
}

void DeepConvNet::save_params(const string& file_name) {
    ofstream ofs(file_name, ios::binary);
    if (!ofs.is_open()) return;

    auto write_string = [&](const string& s) {
        size_t len = s.length();
        ofs.write((char*)&len, sizeof(size_t));
        ofs.write(s.c_str(), len);
        };

    size_t conv_size = params.conv_W.size();
    ofs.write((char*)&conv_size, sizeof(size_t));
    for (auto& pair : params.conv_W) {
        write_string(pair.first);
        int N = pair.second.size(), C = pair.second[0].size(), H = pair.second[0][0].size(), W = pair.second[0][0][0].size();
        ofs.write((char*)&N, sizeof(int)); ofs.write((char*)&C, sizeof(int));
        ofs.write((char*)&H, sizeof(int)); ofs.write((char*)&W, sizeof(int));
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    ofs.write((char*)pair.second[n][c][h].data(), W * sizeof(float));
    }

    size_t affine_size = params.affine_W.size();
    ofs.write((char*)&affine_size, sizeof(size_t));
    for (auto& pair : params.affine_W) {
        write_string(pair.first);
        int R = pair.second.size(), C = pair.second[0].size();
        ofs.write((char*)&R, sizeof(int)); ofs.write((char*)&C, sizeof(int));
        for (int r = 0; r < R; ++r)
            ofs.write((char*)pair.second[r].data(), C * sizeof(float));
    }

    size_t b_size = params.b.size();
    ofs.write((char*)&b_size, sizeof(size_t));
    for (auto& pair : params.b) {
        write_string(pair.first);
        int S = pair.second.size();
        ofs.write((char*)&S, sizeof(int));
        ofs.write((char*)pair.second.data(), S * sizeof(float));
    }
    ofs.close();
}

void DeepConvNet::load_params(const string& file_name) {
    ifstream ifs(file_name, ios::binary);
    if (!ifs.is_open()) return;

    auto read_string = [&]() {
        size_t len;
        ifs.read((char*)&len, sizeof(size_t));
        string s(len, ' ');
        ifs.read(&s[0], len);
        return s;
        };

    size_t conv_size; ifs.read((char*)&conv_size, sizeof(size_t));
    for (size_t i = 0; i < conv_size; ++i) {
        string key = read_string();
        int N, C, H, W;
        ifs.read((char*)&N, sizeof(int)); ifs.read((char*)&C, sizeof(int));
        ifs.read((char*)&H, sizeof(int)); ifs.read((char*)&W, sizeof(int));
        Tensor t(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(W))));
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    ifs.read((char*)t[n][c][h].data(), W * sizeof(float));
        params.conv_W[key] = t;
    }

    size_t affine_size; ifs.read((char*)&affine_size, sizeof(size_t));
    for (size_t i = 0; i < affine_size; ++i) {
        string key = read_string();
        int R, C; ifs.read((char*)&R, sizeof(int)); ifs.read((char*)&C, sizeof(int));
        Matrix m(R, vector<float>(C));
        for (int r = 0; r < R; ++r) ifs.read((char*)m[r].data(), C * sizeof(float));
        params.affine_W[key] = m;
    }

    size_t b_size; ifs.read((char*)&b_size, sizeof(size_t));
    for (size_t i = 0; i < b_size; ++i) {
        string key = read_string();
        int S; ifs.read((char*)&S, sizeof(int));
        vector<float> v(S); ifs.read((char*)v.data(), S * sizeof(float));
        params.b[key] = v;
    }
    ifs.close();
    sync_params_to_layers();
}

void DeepConvNet::sync_params_to_layers() {
    conv1->W = params.conv_W["W1"]; conv1->b = params.b["b1"];
    conv2->W = params.conv_W["W2"]; conv2->b = params.b["b2"];
    conv3->W = params.conv_W["W3"]; conv3->b = params.b["b3"];
    conv4->W = params.conv_W["W4"]; conv4->b = params.b["b4"];
    conv5->W = params.conv_W["W5"]; conv5->b = params.b["b5"];
    conv6->W = params.conv_W["W6"]; conv6->b = params.b["b6"];
    affine1->W = params.affine_W["W7"]; affine1->b = params.b["b7"];
    affine2->W = params.affine_W["W8"]; affine2->b = params.b["b8"];
}