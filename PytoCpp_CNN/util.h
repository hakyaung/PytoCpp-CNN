#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <numeric>

using namespace std;

// 데이터셋(MNIST) 뒤섞는 함수
// x : 훈련 데이터
// t : 정답 레이블
// N : 데이터 개수
// C : 채널 수
// H : 이미지 높이
// W : 이미지 너비
void shuffle_dataset(vector<float>& x, vector<int>& t, int N, int C, int H, int W);

// 합성곱/풀링 계층의 출력 크기 계산 함수
int conv_output_size(int input_size, int filter_size, int stride = 1, int pad = 0);
// 이미지를 입력받아 2차원 배열로 변환 
vector<vector<float>> im2col(const vector<vector<vector<vector<float>>>>& input_data, int filter_h, int filter_w, int stride = 1, int pad = 0);

// 2차원 배열을 이미지로 변환
vector<vector<vector<vector<float>>>> col2im(const vector<vector<float>>& col, int N, int C, int H, int W, int filter_h, int filter_w, int stride = 1, int pad = 0);