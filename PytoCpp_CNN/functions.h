#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

using Matrix = vector<vector<float>>;

// 항등 함수
Matrix identity_function(const Matrix& x);

// 계단 함수
Matrix step_function(const Matrix& x);

// 시그모이드
Matrix sigmoid(const Matrix& x);

// 시그모이드 기울기
Matrix sigmoid_grad(const Matrix& x);

// ReLU
Matrix relu(const Matrix& x);

// ReLU 기울기
Matrix relu_grad(const Matrix& x);

// 소프트맥스
Matrix softmax(const Matrix& x);

// 평균 제곱 오차
float mean_squared_error(const Matrix& y, const Matrix& t);

// 교차 엔트로피
float cross_entropy_error(const Matrix& y, const vector<int>& t);

// softmax loss 
float softmax_loss(const Matrix& X, const vector<int>& t);