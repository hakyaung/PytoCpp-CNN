// 파일 위치: src/train_deepnet.cpp
// 파일 명: train_deepnet.cpp

#include <iostream>
#include <vector>
#include <string>
#include <chrono> // 시간 측정 라이브러리

#include "mnist.h"
#include "deep_convnet.h"
#include "trainer.h"

using namespace std;
using Tensor = vector<vector<vector<vector<float>>>>;

int main() {
    cout << "Loading MNIST dataset..." << endl;

    // 1. MNIST 데이터셋 로드
    MNIST_Dataset dataset = load_mnist(true, false, false);

    // 2. 네트워크 및 트레이너 초기화
    DeepConvNet network;

    Trainer trainer(network,
        dataset.train_img_4d, dataset.train_label,
        dataset.test_img_4d, dataset.test_label,
        1,             // epochs
        100,            // mini_batch_size
        "Adam",         // optimizer
        0.001f,         // lr
        1000,           // evaluate_sample_num_per_epoch
        true);          // verbose

    cout << "Start Training..." << endl;

    // --- 3. 학습 시작 시간 기록 ---
    auto start_time = chrono::high_resolution_clock::now();

    trainer.train();

    // --- 4. 학습 종료 시간 기록 및 소요 시간 계산 ---
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<float> elapsed = end_time - start_time;

    float total_seconds = elapsed.count();
    int minutes = static_cast<int>(total_seconds) / 60;
    float seconds = total_seconds - (minutes * 60);

    cout << "\n========================================" << endl;
    cout << " 학습 완료 소요 시간: " << minutes << "분 " << seconds << "초 (" << total_seconds << "초)" << endl;
    cout << "========================================" << endl;

    // 5. 가중치 저장
    network.save_params("deep_convnet_params.bin");
    cout << "Saved Network Parameters!" << endl;

    return 0;
}