// 파일 위치: PytoCpp_CNN/trainer.cpp
// 파일 명: trainer.cpp

#include <iostream>
#include <random>
#include <algorithm>
#include "trainer.h"
#include "deep_convnet.h"

using namespace std;

// 생성자 구현
Trainer::Trainer(DeepConvNet& network,
    const Tensor& x_train, const vector<int>& t_train,
    const Tensor& x_test, const vector<int>& t_test,
    int epochs, int mini_batch_size,
    string optimizer, float lr,
    int evaluate_sample_num_per_epoch, bool verbose)
    : network(network), x_train(x_train), t_train(t_train),
    x_test(x_test), t_test(t_test)
{
    this->epochs = epochs;
    this->batch_size = mini_batch_size;
    this->evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch;
    this->verbose = verbose;

    this->optimizer_name = optimizer;
    for (char& c : this->optimizer_name) {
        c = tolower(c);
    }

    sgd = SGD(lr);
    momentum = Momentum(lr);
    nesterov = Nesterov(lr);
    adagrad = AdaGrad(lr);
    rmsprop = RMSprop(lr);
    adam = Adam(lr);

    train_size = x_train.size();
    iter_per_epoch = max(train_size / mini_batch_size, 1);
    max_iter = epochs * iter_per_epoch;
    current_iter = 0;
    current_epoch = 0;
}

// train_step 함수 구현
void Trainer::train_step() {
    Tensor x_batch(batch_size);
    vector<int> t_batch(batch_size);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, train_size - 1);

    for (int i = 0; i < batch_size; ++i) {
        int idx = dist(gen);
        x_batch[i] = x_train[idx];
        t_batch[i] = t_train[idx];
    }

    ParamsDict grads = network.gradient(x_batch, t_batch);

    if (optimizer_name == "sgd") sgd.update(network.params, grads);
    else if (optimizer_name == "momentum") momentum.update(network.params, grads);
    else if (optimizer_name == "nesterov") nesterov.update(network.params, grads);
    else if (optimizer_name == "adagrad") adagrad.update(network.params, grads);
    else if (optimizer_name == "rmsprop") rmsprop.update(network.params, grads);
    else if (optimizer_name == "adam") adam.update(network.params, grads);

    network.sync_params_to_layers();

    float loss = network.loss(x_batch, t_batch);
    train_loss_list.push_back(loss);

    if (verbose) {
        cout << "train loss: " << loss << "\n";
    }

    if (current_iter % iter_per_epoch == 0) {
        current_epoch++;

        Tensor x_train_sample, x_test_sample;
        vector<int> t_train_sample, t_test_sample;

        if (evaluate_sample_num_per_epoch > 0) {
            int t_len = evaluate_sample_num_per_epoch;

            x_train_sample.assign(x_train.begin(), x_train.begin() + min(t_len, (int)x_train.size()));
            t_train_sample.assign(t_train.begin(), t_train.begin() + min(t_len, (int)t_train.size()));

            x_test_sample.assign(x_test.begin(), x_test.begin() + min(t_len, (int)x_test.size()));
            t_test_sample.assign(t_test.begin(), t_test.begin() + min(t_len, (int)t_test.size()));
        }
        else {
            x_train_sample = x_train;
            t_train_sample = t_train;
            x_test_sample = x_test;
            t_test_sample = t_test;
        }

        float train_acc = network.accuracy(x_train_sample, t_train_sample);
        float test_acc = network.accuracy(x_test_sample, t_test_sample);

        train_acc_list.push_back(train_acc);
        test_acc_list.push_back(test_acc);

        if (verbose) {
            cout << "=== epoch: " << current_epoch
                << ", train acc: " << train_acc
                << ", test acc: " << test_acc << " ===\n";
        }
    }
    current_iter++;
}

// train 함수 구현
void Trainer::train() {
    for (int i = 0; i < max_iter; ++i) {
        train_step();
    }

    float test_acc = network.accuracy(x_test, t_test);

    if (verbose) {
        cout << "=============== Final Test Accuracy ===============\n";
        cout << "test acc: " << test_acc << "\n";
    }
}