// 파일 명: trainer.h

#pragma once

#include <vector>
#include <string>
#include <map>

#include "util.h"
#include "optimizer.h"

// DeepConvNet 클래스 전방 선언 (포인터 및 참조용)
class DeepConvNet;

using namespace std;

using Matrix = vector<vector<float>>;
using Tensor = vector<vector<vector<vector<float>>>>;

class Trainer {
public:
    // 학습시킬 네트워크 모델의 참조(&)
    DeepConvNet& network;

    bool verbose;
    Tensor x_train;
    vector<int> t_train;
    Tensor x_test;
    vector<int> t_test;

    int epochs;
    int batch_size;
    int evaluate_sample_num_per_epoch;

    string optimizer_name;

    // 옵티마이저 객체들
    SGD sgd;
    Momentum momentum;
    Nesterov nesterov;
    AdaGrad adagrad;
    RMSprop rmsprop;
    Adam adam;

    int train_size;
    int iter_per_epoch;
    int max_iter;
    int current_iter;
    int current_epoch;

    // 평가 기록 리스트
    vector<float> train_loss_list;
    vector<float> train_acc_list;
    vector<float> test_acc_list;

    // 생성자 선언
    Trainer(DeepConvNet& network,
        const Tensor& x_train, const vector<int>& t_train,
        const Tensor& x_test, const vector<int>& t_test,
        int epochs = 20, int mini_batch_size = 100,
        string optimizer = "SGD", float lr = 0.01f,
        int evaluate_sample_num_per_epoch = -1, bool verbose = true);

    // 메서드 선언
    void train_step();
    void train();
};