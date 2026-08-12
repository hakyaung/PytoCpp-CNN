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
void shuffle_dataset(vector<float>& x, vector<int>& t, int N, int C, int H, int W) {
	cout << "데이터셋 뒤섞기!!!\n";

	vector<int> permutation(N);
	for (int i = 0; i < permutation.size(); i++) {
		permutation[i] = i;
	}

	random_device rd;
	mt19937 generator(rd());
	shuffle(permutation.begin(), permutation.end(), generator);

	int image_size = C * H * W;

	vector<float> shuffled_x(x.size());
	vector<int> shuffled_t(t.size());

	for (int i = 0; i < N; i++) {
		int source = permutation[i];
		for (int j = 0; j < image_size; j++) {
			// [i * image_size + 1] -> [i * image_size + j] 로 수정!
			shuffled_x[i * image_size + j] = x[source * image_size + j];
		}
		shuffled_t[i] = t[source];
	}

	x = shuffled_x;
	t = shuffled_t;
}

// 합성곱/풀링 계층의 출력 크기 계산 함수
int conv_output_size(int input_size, int filter_size, int stride = 1, int pad = 0) {
	return (input_size + 2 * pad - filter_size) / stride + 1;
}

// 이미지를 입력받아 2차원 배열로 변환 
vector<vector<float>> im2col(const vector<vector<vector<vector<float>>>>& input_data, int filter_h, int filter_w, int stride = 1, int pad = 0) {
	int N = input_data.size();
	int C = input_data[0].size();
	int H = input_data[0][0].size();
	int W = input_data[0][0][0].size();

	int out_h = (H + 2 * pad - filter_h) / stride + 1;
	int out_w = (W + 2 * pad - filter_w) / stride + 1;

	// Python의 col = np.zeros 한 후 reshape(-1) 한 것과 동일한 행렬
	vector<vector<float>> col(N * out_h * out_w, vector<float>(C * filter_h * filter_w, 0.0f));

	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			for (int fh = 0; fh < filter_h; fh++) {
				for (int fw = 0; fw < filter_w; fw++) {
					for (int oh = 0; oh < out_h; oh++) {
						for (int ow = 0; ow < out_w; ow++) {
							int img_y = oh * stride + fh - pad;
							int img_x = ow * stride + fw - pad;

							if (img_y >= 0 && img_y < H && img_x >= 0 && img_x < W) {
								int row_idx = (n * out_h + oh) * out_w + ow;
								int col_idx = (c * filter_h + fh) * filter_w + fw;

								col[row_idx][col_idx] = input_data[n][c][img_y][img_x];
							}
						}
					}
				}
			}
		}
	}

	return col;
}

// 2차원 배열을 이미지로 변환
vector<vector<vector<vector<float>>>> col2im(const vector<vector<float>>& col, int N, int C, int H, int W, int filter_h, int filter_w, int stride = 1, int pad = 0) {
	int out_h = (H + 2 * pad - filter_h) / stride + 1;
	int out_w = (W + 2 * pad - filter_w) / stride + 1;

	vector<vector<vector<vector<float>>>> img(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(W, 0.0f))));

	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			for (int fh = 0; fh < filter_h; fh++) {
				for (int fw = 0; fw < filter_w; fw++) {
					for (int oh = 0; oh < out_h; oh++) {
						for (int ow = 0; ow < out_w; ow++) {
							int img_y = oh * stride + fh - pad;
							int img_x = ow * stride + fw - pad;

							if (img_y >= 0 && img_y < H && img_x >= 0 && img_x < W) {
								int row_idx = (n * out_h + oh) * out_w + ow;
								int col_idx = (c * filter_h + fh) * filter_w + fw;

								img[n][c][img_y][img_x] += col[row_idx][col_idx];
							}
						}
					}
				}
			}
		}
	}

	return img;
}