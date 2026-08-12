#pragma once

#include <vector>
#include <string>

using namespace std;

using Matrix = vector<vector<float>>;
using Tensor = vector<vector<vector<vector<float>>>>;

// 파이썬의 튜플 반환 ((x_train, t_train), (x_test, t_test))을 대체할 구조체
struct MNIST_Dataset {
    // 1. 이미지 데이터
    Matrix train_img_2d; // flatten = true 일 때 사용
    Tensor train_img_4d; // flatten = false 일 때 사용
    Matrix test_img_2d;
    Tensor test_img_4d;

    // 2. 레이블 데이터
    vector<int> train_label;           // one_hot_label = false 일 때 사용
    Matrix train_label_one_hot;        // one_hot_label = true 일 때 사용
    vector<int> test_label;
    Matrix test_label_one_hot;
};

// 파이썬 원본과 동일한 매개변수 구조
MNIST_Dataset load_mnist(bool normalize = true, bool flatten = true, bool one_hot_label = false);