// Combined single-file build of the project files in PytoCpp_CNN
// This file concatenates headers and implementations into one translation unit.

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <random>
#include <numeric>
#include <cstdlib>
#include <cstdint>
#include <map>

using namespace std;

using Matrix = vector<vector<float>>;
using Tensor = vector<vector<vector<vector<float>>>>;

// Forward declarations / shared types

// util functions
void shuffle_dataset(vector<float>& x, vector<int>& t, int N, int C, int H, int W);
int conv_output_size(int input_size, int filter_size, int stride = 1, int pad = 0);
vector<vector<float>> im2col(const vector<vector<vector<vector<float>>>>& input_data, int filter_h, int filter_w, int stride = 1, int pad = 0);
vector<vector<vector<vector<float>>>> col2im(const vector<vector<float>>& col, int N, int C, int H, int W, int filter_h, int filter_w, int stride = 1, int pad = 0);

// functions (activation / loss helpers)
Matrix identity_function(const Matrix& x);
Matrix step_function(const Matrix& x);
Matrix sigmoid(const Matrix& x);
Matrix sigmoid_grad(const Matrix& x);
Matrix relu(const Matrix& x);
Matrix relu_grad(const Matrix& x);
Matrix softmax(const Matrix& x);
float mean_squared_error(const Matrix& y, const Matrix& t);
float cross_entropy_error(const Matrix& y, const vector<int>& t);
float softmax_loss(const Matrix& X, const vector<int>& t);

// ParamsDict used by optimizers and network
struct ParamsDict {
	map<string, Tensor> conv_W;
	map<string, Matrix> affine_W;
	map<string, vector<float>> b;
	bool empty() const {
		return conv_W.empty() && affine_W.empty() && b.empty();
	}
};

// Optimizers
class SGD { public: float lr; SGD(float lr = 0.01f); void update(ParamsDict& params, ParamsDict& grads); };
class Momentum { public: float lr; float momentum; ParamsDict v; Momentum(float lr = 0.01f, float momentum = 0.9f); void update(ParamsDict& params, ParamsDict& grads); };
class Nesterov { public: float lr; float momentum; ParamsDict v; Nesterov(float lr = 0.01f, float momentum = 0.9f); void update(ParamsDict& params, ParamsDict& grads); };
class AdaGrad { public: float lr; ParamsDict h; AdaGrad(float lr = 0.01f); void update(ParamsDict& params, ParamsDict& grads); };
class RMSprop { public: float lr; float decay_rate; ParamsDict h; RMSprop(float lr = 0.01f, float decay_rate = 0.99f); void update(ParamsDict& params, ParamsDict& grads); };
class Adam { public: float lr; float beta1; float beta2; int iter; ParamsDict m; ParamsDict v; Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f); void update(ParamsDict& params, ParamsDict& grads); };

// Layers
class Relu { public: Tensor mask; Relu(); Tensor forward(const Tensor& x); Tensor backward(const Tensor& dout); };
class Sigmoid { public: Tensor out_val; Sigmoid(); Tensor forward(const Tensor& x); Tensor backward(const Tensor& dout); };
class Affine { public: Matrix W; vector<float> b; Matrix x_mat; vector<int> original_x_shape; Matrix dW; vector<float> db; Affine(const Matrix& W_init, const vector<float>& b_init); Matrix forward(const Tensor& x_in); Tensor backward(const Matrix& dout); };
class SoftmaxLoss { public: float loss_val; Matrix y; vector<int> t; SoftmaxLoss(); float forward(const Matrix& x, const vector<int>& t_in); Matrix backward(float dout = 1.0f); };
class Dropout { public: float dropout_ratio; Tensor mask; Dropout(float ratio = 0.5f); Tensor forward(const Tensor& x, bool train_flg = true); Tensor backward(const Tensor& dout); };
class Convolution { public: Tensor W; vector<float> b; int stride, pad; Tensor x; Matrix col, col_W; Tensor dW; vector<float> db; Convolution(const Tensor& W_init, const vector<float>& b_init, int stride = 1, int pad = 0); Tensor forward(const Tensor& x_in); Tensor backward(const Tensor& dout); };
class Pooling { public: int pool_h, pool_w, stride, pad; Tensor x; vector<int> arg_max; Pooling(int pool_h, int pool_w, int stride = 2, int pad = 0); Tensor forward(const Tensor& x_in); Tensor backward(const Tensor& dout); };

// MNIST dataset structure and loader
struct MNIST_Dataset {
	Matrix train_img_2d; Tensor train_img_4d; Matrix test_img_2d; Tensor test_img_4d;
	vector<int> train_label; Matrix train_label_one_hot; vector<int> test_label; Matrix test_label_one_hot;
};
MNIST_Dataset load_mnist(bool normalize = true, bool flatten = true, bool one_hot_label = false);

// DeepConvNet declaration
class DeepConvNet {
public:
	ParamsDict params;
	Convolution* conv1; Convolution* conv2; Convolution* conv3;
	Convolution* conv4; Convolution* conv5; Convolution* conv6;
	Relu* relu1; Relu* relu2; Relu* relu3; Relu* relu4; Relu* relu5; Relu* relu6; Relu* relu7;
	Pooling* pool1; Pooling* pool2; Pooling* pool3;
	Affine* affine1; Affine* affine2;
	Dropout* drop1; Dropout* drop2;
	SoftmaxLoss* last_layer;
	DeepConvNet(); ~DeepConvNet();
	Matrix predict(Tensor x, bool train_flg = false);
	float loss(const Tensor& x, const vector<int>& t);
	float accuracy(const Tensor& x, const vector<int>& t, int batch_size = 100);
	ParamsDict gradient(const Tensor& x, const vector<int>& t);
	void save_params(const string& file_name = "params.bin");
	void load_params(const string& file_name = "params.bin");
	void sync_params_to_layers();
};

// Trainer
class Trainer {
public:
	DeepConvNet& network;
	bool verbose; Tensor x_train; vector<int> t_train; Tensor x_test; vector<int> t_test;
	int epochs; int batch_size; int evaluate_sample_num_per_epoch; string optimizer_name;
	SGD sgd; Momentum momentum; Nesterov nesterov; AdaGrad adagrad; RMSprop rmsprop; Adam adam;
	int train_size; int iter_per_epoch; int max_iter; int current_iter; int current_epoch;
	vector<float> train_loss_list; vector<float> train_acc_list; vector<float> test_acc_list;
	Trainer(DeepConvNet& network, const Tensor& x_train, const vector<int>& t_train, const Tensor& x_test, const vector<int>& t_test, int epochs = 20, int mini_batch_size = 100, string optimizer = "SGD", float lr = 0.01f, int evaluate_sample_num_per_epoch = -1, bool verbose = true);
	void train_step(); void train();
};

// ==================================================
// Implementations (from source files)
// ==================================================

// ---------- functions.cpp ----------
Matrix identity_function(const Matrix& x) { return x; }

Matrix step_function(const Matrix& x) {
	int N = x.size(), D = x[0].size();
	Matrix out(N, vector<float>(D, 0.0f));
	for (int i = 0; i < N; ++i) for (int j = 0; j < D; ++j) out[i][j] = (x[i][j] > 0.0f) ? 1.0f : 0.0f;
	return out;
}

Matrix sigmoid(const Matrix& x) {
	int N = x.size(), D = x[0].size(); Matrix out(N, vector<float>(D, 0.0f));
	for (int i = 0; i < N; ++i) for (int j = 0; j < D; ++j) out[i][j] = 1.0f / (1.0f + exp(-x[i][j]));
	return out;
}

Matrix sigmoid_grad(const Matrix& x) { Matrix s = sigmoid(x); int N = s.size(), D = s[0].size(); Matrix grad(N, vector<float>(D, 0.0f)); for (int i=0;i<N;++i) for(int j=0;j<D;++j) grad[i][j] = (1.0f - s[i][j]) * s[i][j]; return grad; }

Matrix relu(const Matrix& x) { int N=x.size(), D=x[0].size(); Matrix out(N, vector<float>(D,0.0f)); for(int i=0;i<N;++i) for(int j=0;j<D;++j) out[i][j] = max(0.0f, x[i][j]); return out; }

Matrix relu_grad(const Matrix& x) { int N=x.size(), D=x[0].size(); Matrix grad(N, vector<float>(D,0.0f)); for(int i=0;i<N;++i) for(int j=0;j<D;++j) grad[i][j] = (x[i][j] >= 0.0f) ? 1.0f : 0.0f; return grad; }

Matrix softmax(const Matrix& x) {
	int N = x.size(), C = x[0].size(); Matrix y(N, vector<float>(C,0.0f));
	for (int i=0;i<N;++i) {
		float max_val = x[i][0]; for (int j=1;j<C;++j) if (x[i][j] > max_val) max_val = x[i][j];
		float sum_exp = 0.0f; for (int j=0;j<C;++j) { y[i][j] = exp(x[i][j] - max_val); sum_exp += y[i][j]; }
		for (int j=0;j<C;++j) y[i][j] /= sum_exp;
	}
	return y;
}

float mean_squared_error(const Matrix& y, const Matrix& t) { int N=y.size(), C=y[0].size(); float sum=0.0f; for(int i=0;i<N;++i) for(int j=0;j<C;++j){ float diff = y[i][j]-t[i][j]; sum += diff*diff;} return 0.5f * sum; }

float cross_entropy_error(const Matrix& y, const vector<int>& t) { int batch_size = y.size(); float sum=0.0f; for (int i=0;i<batch_size;++i) sum -= log(y[i][t[i]] + 1e-7f); return sum / batch_size; }

float softmax_loss(const Matrix& X, const vector<int>& t) { Matrix y = softmax(X); return cross_entropy_error(y, t); }

// ---------- util.cpp ----------
void shuffle_dataset(vector<float>& x, vector<int>& t, int N, int C, int H, int W) {
	cout << "데이터셋 뒤섞기!!!";
	vector<int> permutation(N);
	for (int i = 0; i < permutation.size(); i++) permutation[i] = i;
	random_device rd; mt19937 generator(rd()); shuffle(permutation.begin(), permutation.end(), generator);
	int image_size = C * H * W;
	vector<float> shuffled_x(x.size()); vector<int> shuffled_t(t.size());
	for (int i = 0; i < N; i++) {
		int source = permutation[i];
		for (int j = 0; j < image_size; j++) {
			// fixed index: use j not 1
			shuffled_x[i * image_size + j] = x[source * image_size + j];
		}
		shuffled_t[i] = t[source];
	}
	x = shuffled_x; t = shuffled_t;
}

int conv_output_size(int input_size, int filter_size, int stride, int pad) { return (input_size + 2 * pad - filter_size) / stride + 1; }

vector<vector<float>> im2col(const vector<vector<vector<vector<float>>>>& input_data, int filter_h, int filter_w, int stride, int pad) {
	int N = input_data.size(); int C = input_data[0].size(); int H = input_data[0][0].size(); int W = input_data[0][0][0].size();
	int out_h = (H + 2 * pad - filter_h) / stride + 1; int out_w = (W + 2 * pad - filter_w) / stride + 1;
	vector<vector<float>> col(N * out_h * out_w, vector<float>(C * filter_h * filter_w, 0.0f));
	for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int fh=0;fh<filter_h;++fh) for (int fw=0;fw<filter_w;++fw) for (int oh=0;oh<out_h;++oh) for (int ow=0;ow<out_w;++ow) {
		int img_y = oh * stride + fh - pad; int img_x = ow * stride + fw - pad;
		if (img_y >= 0 && img_y < H && img_x >= 0 && img_x < W) {
			int row_idx = (n * out_h + oh) * out_w + ow; int col_idx = (c * filter_h + fh) * filter_w + fw;
			col[row_idx][col_idx] = input_data[n][c][img_y][img_x];
		}
	}
	return col;
}

vector<vector<vector<vector<float>>>> col2im(const vector<vector<float>>& col, int N, int C, int H, int W, int filter_h, int filter_w, int stride, int pad) {
	int out_h = (H + 2 * pad - filter_h) / stride + 1; int out_w = (W + 2 * pad - filter_w) / stride + 1;
	vector<vector<vector<vector<float>>>> img(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(W, 0.0f))));
	for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int fh=0;fh<filter_h;++fh) for (int fw=0;fw<filter_w;++fw) for (int oh=0;oh<out_h;++oh) for (int ow=0;ow<out_w;++ow) {
		int img_y = oh * stride + fh - pad; int img_x = ow * stride + fw - pad;
		if (img_y >= 0 && img_y < H && img_x >= 0 && img_x < W) {
			int row_idx = (n * out_h + oh) * out_w + ow; int col_idx = (c * filter_h + fh) * filter_w + fw;
			img[n][c][img_y][img_x] += col[row_idx][col_idx];
		}
	}
	return img;
}

// ---------- layers.cpp ----------
Relu::Relu() {}
Tensor Relu::forward(const Tensor& x) {
	int N=x.size(), C=x[0].size(), H=x[0][0].size(), Wdim=x[0][0][0].size();
	mask = Tensor(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(Wdim,0.0f))));
	Tensor out = x;
	for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int h=0;h<H;++h) for (int w=0;w<Wdim;++w) if (x[n][c][h][w] <= 0) { mask[n][c][h][w] = 1.0f; out[n][c][h][w] = 0.0f; }
	return out;
}
Tensor Relu::backward(const Tensor& dout) {
	int N=dout.size(), C=dout[0].size(), H=dout[0][0].size(), Wdim=dout[0][0][0].size(); Tensor dx = dout; for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int h=0;h<H;++h) for (int w=0;w<Wdim;++w) if (mask[n][c][h][w] == 1.0f) dx[n][c][h][w] = 0.0f; return dx;
}

Sigmoid::Sigmoid() {}
Tensor Sigmoid::forward(const Tensor& x) { int N=x.size(), C=x[0].size(), H=x[0][0].size(), Wdim=x[0][0][0].size(); Tensor out = x; for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int h=0;h<H;++h) for (int w=0;w<Wdim;++w) out[n][c][h][w] = 1.0f / (1.0f + exp(-x[n][c][h][w])); out_val = out; return out; }
Tensor Sigmoid::backward(const Tensor& dout) { int N=dout.size(), C=dout[0].size(), H=dout[0][0].size(), Wdim=dout[0][0][0].size(); Tensor dx = dout; for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int h=0;h<H;++h) for (int w=0;w<Wdim;++w) dx[n][c][h][w] = dout[n][c][h][w] * (1.0f - out_val[n][c][h][w]); return dx; }

Affine::Affine(const Matrix& W_init, const vector<float>& b_init) { this->W = W_init; this->b = b_init; }
Matrix Affine::forward(const Tensor& x_in) {
	int N=x_in.size(); int C=x_in[0].size(); int H=x_in[0][0].size(); int Wdim=x_in[0][0][0].size(); original_x_shape = {N,C,H,Wdim}; int D = C*H*Wdim; x_mat = Matrix(N, vector<float>(D,0.0f)); for (int n=0;n<N;++n){ int idx=0; for (int c=0;c<C;++c) for (int h=0;h<H;++h) for (int w=0;w<Wdim;++w) x_mat[n][idx++] = x_in[n][c][h][w]; }
	Matrix out(N, vector<float>(W[0].size(),0.0f)); for (int n=0;n<N;++n) for (size_t j=0;j<W[0].size();++j){ for (size_t k=0;k<x_mat[0].size();++k) out[n][j] += x_mat[n][k] * W[k][j]; out[n][j] += b[j]; }
	return out;
}
Tensor Affine::backward(const Matrix& dout) {
	int N=dout.size(), D=W.size(), out_size=W[0].size(); Matrix dx_mat(N, vector<float>(D,0.0f)); for (int n=0;n<N;++n) for (int j=0;j<D;++j) for (int k=0;k<out_size;++k) dx_mat[n][j] += dout[n][k] * W[j][k];
	dW = Matrix(D, vector<float>(out_size,0.0f)); for (int i=0;i<D;++i) for (int j=0;j<out_size;++j) for (int k=0;k<N;++k) dW[i][j] += x_mat[k][i] * dout[k][j];
	db = vector<float>(out_size,0.0f); for (int d=0; d<out_size; ++d) for (int n=0;n<N;++n) db[d] += dout[n][d];
	Tensor dx(original_x_shape[0], vector<vector<vector<float>>>(original_x_shape[1], vector<vector<float>>(original_x_shape[2], vector<float>(original_x_shape[3])))); for (int n=0;n<N;++n){ int idx=0; for (int c=0;c<original_x_shape[1];++c) for (int h=0;h<original_x_shape[2];++h) for (int w=0;w<original_x_shape[3];++w) dx[n][c][h][w] = dx_mat[n][idx++]; }
	return dx;
}

SoftmaxLoss::SoftmaxLoss() {}
float SoftmaxLoss::forward(const Matrix& x, const vector<int>& t_in) { t = t_in; int N=x.size(); int C=x[0].size(); y = Matrix(N, vector<float>(C,0.0f)); for (int n=0;n<N;++n){ float max_val = x[n][0]; for (int c=1;c<C;++c) if (x[n][c] > max_val) max_val = x[n][c]; float sum_exp = 0.0f; for (int c=0;c<C;++c){ y[n][c] = exp(x[n][c] - max_val); sum_exp += y[n][c]; } for (int c=0;c<C;++c) y[n][c] /= sum_exp; } loss_val = 0.0f; for (int n=0;n<N;++n) { loss_val -= log(y[n][t[n]] + 1e-7f); } loss_val /= N; return loss_val; }
Matrix SoftmaxLoss::backward(float dout) { int batch_size = t.size(); int C = y[0].size(); Matrix dx = y; for (int n=0;n<batch_size;++n){ dx[n][t[n]] -= 1.0f; for (int c=0;c<C;++c) dx[n][c] = (dx[n][c] * dout) / batch_size; } return dx; }

Dropout::Dropout(float ratio) { dropout_ratio = ratio; }
Tensor Dropout::forward(const Tensor& x, bool train_flg) { int N=x.size(), C=x[0].size(), H=x[0][0].size(), Wdim=x[0][0][0].size(); Tensor out = x; if (train_flg) { mask = Tensor(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(Wdim,0.0f)))); for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int h=0;h<H;++h) for (int w=0;w<Wdim;++w) { float rand_val = static_cast<float>(rand()) / RAND_MAX; if (rand_val > dropout_ratio) mask[n][c][h][w] = 1.0f; else { mask[n][c][h][w] = 0.0f; out[n][c][h][w] = 0.0f; } } return out; } else { for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int h=0;h<H;++h) for (int w=0;w<Wdim;++w) out[n][c][h][w] = x[n][c][h][w] * (1.0f - dropout_ratio); return out; } }
Tensor Dropout::backward(const Tensor& dout) { int N=dout.size(), C=dout[0].size(), H=dout[0][0].size(), Wdim=dout[0][0][0].size(); Tensor dx = dout; for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int h=0;h<H;++h) for (int w=0;w<Wdim;++w) dx[n][c][h][w] = dout[n][c][h][w] * mask[n][c][h][w]; return dx; }

Convolution::Convolution(const Tensor& W_init, const vector<float>& b_init, int stride, int pad) { this->W = W_init; this->b = b_init; this->stride = stride; this->pad = pad; }
Tensor Convolution::forward(const Tensor& x_in) {
	int FN=W.size(), C=W[0].size(), FH=W[0][0].size(), FW=W[0][0][0].size(); int N=x_in.size(), H=x_in[0][0].size(), Wdim=x_in[0][0][0].size(); int out_h = 1 + (H + 2 * pad - FH) / stride; int out_w = 1 + (Wdim + 2 * pad - FW) / stride; col = im2col(x_in, FH, FW, stride, pad);
	col_W = Matrix(C * FH * FW, vector<float>(FN, 0.0f)); for (int fn=0;fn<FN;++fn){ int idx=0; for (int c=0;c<C;++c) for (int fh=0;fh<FH;++fh) for (int fw=0;fw<FW;++fw) col_W[idx++][fn] = W[fn][c][fh][fw]; }
	Matrix out_mat(col.size(), vector<float>(col_W[0].size(), 0.0f)); for (size_t i=0;i<col.size();++i) for (size_t j=0;j<col_W[0].size();++j){ for (size_t k=0;k<col[0].size();++k) out_mat[i][j] += col[i][k] * col_W[k][j]; out_mat[i][j] += b[j]; }
	Tensor out(N, vector<vector<vector<float>>>(FN, vector<vector<float>>(out_h, vector<float>(out_w)))); for (int n=0;n<N;++n) for (int oh=0;oh<out_h;++oh) for (int ow=0;ow<out_w;++ow){ int row_idx = (n * out_h + oh) * out_w + ow; for (int fn=0;fn<FN;++fn) out[n][fn][oh][ow] = out_mat[row_idx][fn]; }
	x = x_in; return out;
}
Tensor Convolution::backward(const Tensor& dout) {
	int FN=W.size(), C=W[0].size(), FH=W[0][0].size(), FW=W[0][0][0].size(); int N=dout.size(), out_h=dout[0][0].size(), out_w=dout[0][0][0].size(); Matrix dout_mat(N * out_h * out_w, vector<float>(FN,0.0f)); for (int n=0;n<N;++n) for (int oh=0;oh<out_h;++oh) for (int ow=0;ow<out_w;++ow){ int row_idx = (n * out_h + oh) * out_w + ow; for (int fn=0;fn<FN;++fn) dout_mat[row_idx][fn] = dout[n][fn][oh][ow]; }
	db = vector<float>(FN,0.0f); for (size_t i=0;i<dout_mat.size();++i) for (int fn=0;fn<FN;++fn) db[fn] += dout_mat[i][fn];
	Matrix dW_mat(col[0].size(), vector<float>(dout_mat[0].size(), 0.0f)); for (size_t i=0;i<col[0].size();++i) for (size_t j=0;j<dout_mat[0].size();++j) for (size_t k=0;k<col.size();++k) dW_mat[i][j] += col[k][i] * dout_mat[k][j];
	dW = Tensor(FN, vector<vector<vector<float>>>(C, vector<vector<float>>(FH, vector<float>(FW)))); for (int fn=0;fn<FN;++fn){ int idx=0; for (int c=0;c<C;++c) for (int fh=0;fh<FH;++fh) for (int fw=0;fw<FW;++fw) dW[fn][c][fh][fw] = dW_mat[idx++][fn]; }
	Matrix dcol(dout_mat.size(), vector<float>(col_W.size(), 0.0f)); for (size_t i=0;i<dout_mat.size();++i) for (size_t j=0;j<col_W.size();++j) for (size_t k=0;k<dout_mat[0].size();++k) dcol[i][j] += dout_mat[i][k] * col_W[j][k];
	Tensor dx = col2im(dcol, N, x[0].size(), x[0][0].size(), x[0][0][0].size(), FH, FW, stride, pad);
	return dx;
}

Pooling::Pooling(int pool_h, int pool_w, int stride, int pad) { this->pool_h = pool_h; this->pool_w = pool_w; this->stride = stride; this->pad = pad; }
Tensor Pooling::forward(const Tensor& x_in) {
	int N=x_in.size(), C=x_in[0].size(), H=x_in[0][0].size(), Wdim=x_in[0][0][0].size(); int out_h = 1 + (H - pool_h) / stride; int out_w = 1 + (Wdim - pool_w) / stride;
	Matrix col = im2col(x_in, pool_h, pool_w, stride, pad); int col_rows = col.size(); int col_cols = col[0].size(); arg_max = vector<int>(col_rows); vector<float> out_flat(col_rows);
	for (int i=0;i<col_rows;++i){ float max_val = col[i][0]; int max_idx = 0; for (int j=1;j<col_cols;++j) if (col[i][j] > max_val) { max_val = col[i][j]; max_idx = j; } out_flat[i] = max_val; arg_max[i] = max_idx; }
	Tensor out(N, vector<vector<vector<float>>>(C, vector<vector<float>>(out_h, vector<float>(out_w)))); for (int n=0;n<N;++n) for (int oh=0;oh<out_h;++oh) for (int ow=0;ow<out_w;++ow) for (int c=0;c<C;++c){ int row_idx = (n * C + c) * out_h * out_w + (oh * out_w + ow); out[n][c][oh][ow] = out_flat[row_idx]; }
	x = x_in; return out;
}
Tensor Pooling::backward(const Tensor& dout) {
	int N=dout.size(), C=dout[0].size(), out_h=dout[0][0].size(), out_w=dout[0][0][0].size(); int pool_size = pool_h * pool_w; int col_rows = arg_max.size(); Matrix dmax(col_rows, vector<float>(pool_size, 0.0f));
	for (int n=0;n<N;++n) for (int oh=0;oh<out_h;++oh) for (int ow=0;ow<out_w;++ow) for (int c=0;c<C;++c){ int row_idx = (n * C + c) * out_h * out_w + (oh * out_w + ow); dmax[row_idx][arg_max[row_idx]] = dout[n][c][oh][ow]; }
	Tensor dx = col2im(dmax, N, C, x[0][0].size(), x[0][0][0].size(), pool_h, pool_w, stride, pad); return dx;
}

// ---------- optimizer.cpp ----------
SGD::SGD(float lr) { this->lr = lr; }
void SGD::update(ParamsDict& params, ParamsDict& grads) {
	for (auto& pair : params.conv_W) { string key = pair.first; for (size_t i = 0; i < params.conv_W[key].size(); ++i) for (size_t j = 0; j < params.conv_W[key][0].size(); ++j) for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k) for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l) params.conv_W[key][i][j][k][l] -= lr * grads.conv_W[key][i][j][k][l]; }
	for (auto& pair : params.affine_W) { string key = pair.first; for (size_t i = 0; i < params.affine_W[key].size(); ++i) for (size_t j = 0; j < params.affine_W[key][0].size(); ++j) params.affine_W[key][i][j] -= lr * grads.affine_W[key][i][j]; }
	for (auto& pair : params.b) { string key = pair.first; for (size_t i = 0; i < params.b[key].size(); ++i) params.b[key][i] -= lr * grads.b[key][i]; }
}

Momentum::Momentum(float lr, float momentum) { this->lr = lr; this->momentum = momentum; }
void Momentum::update(ParamsDict& params, ParamsDict& grads) {
	if (v.empty()) {
		for (auto& pair : params.conv_W) { auto& p = pair.second; v.conv_W[pair.first] = Tensor(p.size(), vector<vector<vector<float>>>(p[0].size(), vector<vector<float>>(p[0][0].size(), vector<float>(p[0][0][0].size(), 0.0f)))); }
		for (auto& pair : params.affine_W) { auto& p = pair.second; v.affine_W[pair.first] = Matrix(p.size(), vector<float>(p[0].size(), 0.0f)); }
		for (auto& pair : params.b) { v.b[pair.first] = vector<float>(pair.second.size(), 0.0f); }
	}
	for (auto& pair : params.conv_W) { string key = pair.first; for (size_t i = 0; i < params.conv_W[key].size(); ++i) for (size_t j = 0; j < params.conv_W[key][0].size(); ++j) for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k) for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l) { v.conv_W[key][i][j][k][l] = momentum * v.conv_W[key][i][j][k][l] - lr * grads.conv_W[key][i][j][k][l]; params.conv_W[key][i][j][k][l] += v.conv_W[key][i][j][k][l]; } }
	for (auto& pair : params.affine_W) { string key = pair.first; for (size_t i = 0; i < params.affine_W[key].size(); ++i) for (size_t j = 0; j < params.affine_W[key][0].size(); ++j) { v.affine_W[key][i][j] = momentum * v.affine_W[key][i][j] - lr * grads.affine_W[key][i][j]; params.affine_W[key][i][j] += v.affine_W[key][i][j]; } }
	for (auto& pair : params.b) { string key = pair.first; for (size_t i = 0; i < params.b[key].size(); ++i) { v.b[key][i] = momentum * v.b[key][i] - lr * grads.b[key][i]; params.b[key][i] += v.b[key][i]; } }
}

Nesterov::Nesterov(float lr, float momentum) { this->lr = lr; this->momentum = momentum; }
void Nesterov::update(ParamsDict& params, ParamsDict& grads) {
	if (v.empty()) { for (auto& pair : params.conv_W) { auto& p = pair.second; v.conv_W[pair.first] = Tensor(p.size(), vector<vector<vector<float>>>(p[0].size(), vector<vector<float>>(p[0][0].size(), vector<float>(p[0][0][0].size(), 0.0f)))); } for (auto& pair : params.affine_W) { auto& p = pair.second; v.affine_W[pair.first] = Matrix(p.size(), vector<float>(p[0].size(), 0.0f)); } for (auto& pair : params.b) { v.b[pair.first] = vector<float>(pair.second.size(), 0.0f); } }
	for (auto& pair : params.conv_W) { string key = pair.first; for (size_t i = 0; i < params.conv_W[key].size(); ++i) for (size_t j = 0; j < params.conv_W[key][0].size(); ++j) for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k) for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l) { v.conv_W[key][i][j][k][l] *= momentum; v.conv_W[key][i][j][k][l] -= lr * grads.conv_W[key][i][j][k][l]; params.conv_W[key][i][j][k][l] += momentum * momentum * v.conv_W[key][i][j][k][l]; params.conv_W[key][i][j][k][l] -= (1.0f + momentum) * lr * grads.conv_W[key][i][j][k][l]; } }
	for (auto& pair : params.affine_W) { string key = pair.first; for (size_t i = 0; i < params.affine_W[key].size(); ++i) for (size_t j = 0; j < params.affine_W[key][0].size(); ++j) { v.affine_W[key][i][j] *= momentum; v.affine_W[key][i][j] -= lr * grads.affine_W[key][i][j]; params.affine_W[key][i][j] += momentum * momentum * v.affine_W[key][i][j]; params.affine_W[key][i][j] -= (1.0f + momentum) * lr * grads.affine_W[key][i][j]; } }
	for (auto& pair : params.b) { string key = pair.first; for (size_t i = 0; i < params.b[key].size(); ++i) { v.b[key][i] *= momentum; v.b[key][i] -= lr * grads.b[key][i]; params.b[key][i] += momentum * momentum * v.b[key][i]; params.b[key][i] -= (1.0f + momentum) * lr * grads.b[key][i]; } }
}

AdaGrad::AdaGrad(float lr) { this->lr = lr; }
void AdaGrad::update(ParamsDict& params, ParamsDict& grads) {
	if (h.empty()) { for (auto& pair : params.conv_W) { auto& p = pair.second; h.conv_W[pair.first] = Tensor(p.size(), vector<vector<vector<float>>>(p[0].size(), vector<vector<float>>(p[0][0].size(), vector<float>(p[0][0][0].size(), 0.0f)))); } for (auto& pair : params.affine_W) { auto& p = pair.second; h.affine_W[pair.first] = Matrix(p.size(), vector<float>(p[0].size(), 0.0f)); } for (auto& pair : params.b) { h.b[pair.first] = vector<float>(pair.second.size(), 0.0f); } }
	for (auto& pair : params.conv_W) { string key = pair.first; for (size_t i = 0; i < params.conv_W[key].size(); ++i) for (size_t j = 0; j < params.conv_W[key][0].size(); ++j) for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k) for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l) { h.conv_W[key][i][j][k][l] += grads.conv_W[key][i][j][k][l] * grads.conv_W[key][i][j][k][l]; params.conv_W[key][i][j][k][l] -= lr * grads.conv_W[key][i][j][k][l] / (sqrt(h.conv_W[key][i][j][k][l]) + 1e-7f); } }
	for (auto& pair : params.affine_W) { string key = pair.first; for (size_t i = 0; i < params.affine_W[key].size(); ++i) for (size_t j = 0; j < params.affine_W[key][0].size(); ++j) { h.affine_W[key][i][j] += grads.affine_W[key][i][j] * grads.affine_W[key][i][j]; params.affine_W[key][i][j] -= lr * grads.affine_W[key][i][j] / (sqrt(h.affine_W[key][i][j]) + 1e-7f); } }
	for (auto& pair : params.b) { string key = pair.first; for (size_t i = 0; i < params.b[key].size(); ++i) { h.b[key][i] += grads.b[key][i] * grads.b[key][i]; params.b[key][i] -= lr * grads.b[key][i] / (sqrt(h.b[key][i]) + 1e-7f); } }
}

RMSprop::RMSprop(float lr, float decay_rate) { this->lr = lr; this->decay_rate = decay_rate; }
void RMSprop::update(ParamsDict& params, ParamsDict& grads) {
	if (h.empty()) { for (auto& pair : params.conv_W) { auto& p = pair.second; h.conv_W[pair.first] = Tensor(p.size(), vector<vector<vector<float>>>(p[0].size(), vector<vector<float>>(p[0][0].size(), vector<float>(p[0][0][0].size(), 0.0f)))); } for (auto& pair : params.affine_W) { auto& p = pair.second; h.affine_W[pair.first] = Matrix(p.size(), vector<float>(p[0].size(), 0.0f)); } for (auto& pair : params.b) { h.b[pair.first] = vector<float>(pair.second.size(), 0.0f); } }
	for (auto& pair : params.conv_W) { string key = pair.first; for (size_t i = 0; i < params.conv_W[key].size(); ++i) for (size_t j = 0; j < params.conv_W[key][0].size(); ++j) for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k) for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l) { h.conv_W[key][i][j][k][l] *= decay_rate; h.conv_W[key][i][j][k][l] += (1.0f - decay_rate) * grads.conv_W[key][i][j][k][l] * grads.conv_W[key][i][j][k][l]; params.conv_W[key][i][j][k][l] -= lr * grads.conv_W[key][i][j][k][l] / (sqrt(h.conv_W[key][i][j][k][l]) + 1e-7f); } }
	for (auto& pair : params.affine_W) { string key = pair.first; for (size_t i = 0; i < params.affine_W[key].size(); ++i) for (size_t j = 0; j < params.affine_W[key][0].size(); ++j) { h.affine_W[key][i][j] *= decay_rate; h.affine_W[key][i][j] += (1.0f - decay_rate) * grads.affine_W[key][i][j] * grads.affine_W[key][i][j]; params.affine_W[key][i][j] -= lr * grads.affine_W[key][i][j] / (sqrt(h.affine_W[key][i][j]) + 1e-7f); } }
	for (auto& pair : params.b) { string key = pair.first; for (size_t i = 0; i < params.b[key].size(); ++i) { h.b[key][i] *= decay_rate; h.b[key][i] += (1.0f - decay_rate) * grads.b[key][i] * grads.b[key][i]; params.b[key][i] -= lr * grads.b[key][i] / (sqrt(h.b[key][i]) + 1e-7f); } }
}

Adam::Adam(float lr, float beta1, float beta2) { this->lr = lr; this->beta1 = beta1; this->beta2 = beta2; this->iter = 0; }
void Adam::update(ParamsDict& params, ParamsDict& grads) {
	if (m.empty()) { for (auto& pair : params.conv_W) { auto& p = pair.second; m.conv_W[pair.first] = Tensor(p.size(), vector<vector<vector<float>>>(p[0].size(), vector<vector<float>>(p[0][0].size(), vector<float>(p[0][0][0].size(), 0.0f)))); v.conv_W[pair.first] = m.conv_W[pair.first]; } for (auto& pair : params.affine_W) { auto& p = pair.second; m.affine_W[pair.first] = Matrix(p.size(), vector<float>(p[0].size(), 0.0f)); v.affine_W[pair.first] = m.affine_W[pair.first]; } for (auto& pair : params.b) { m.b[pair.first] = vector<float>(pair.second.size(), 0.0f); v.b[pair.first] = m.b[pair.first]; } }
	iter++; float lr_t = lr * sqrt(1.0f - pow(beta2, iter)) / (1.0f - pow(beta1, iter));
	for (auto& pair : params.conv_W) { string key = pair.first; for (size_t i = 0; i < params.conv_W[key].size(); ++i) for (size_t j = 0; j < params.conv_W[key][0].size(); ++j) for (size_t k = 0; k < params.conv_W[key][0][0].size(); ++k) for (size_t l = 0; l < params.conv_W[key][0][0][0].size(); ++l) { m.conv_W[key][i][j][k][l] += (1.0f - beta1) * (grads.conv_W[key][i][j][k][l] - m.conv_W[key][i][j][k][l]); v.conv_W[key][i][j][k][l] += (1.0f - beta2) * (grads.conv_W[key][i][j][k][l] * grads.conv_W[key][i][j][k][l] - v.conv_W[key][i][j][k][l]); params.conv_W[key][i][j][k][l] -= lr_t * m.conv_W[key][i][j][k][l] / (sqrt(v.conv_W[key][i][j][k][l]) + 1e-7f); } }
	for (auto& pair : params.affine_W) { string key = pair.first; for (size_t i = 0; i < params.affine_W[key].size(); ++i) for (size_t j = 0; j < params.affine_W[key][0].size(); ++j) { m.affine_W[key][i][j] += (1.0f - beta1) * (grads.affine_W[key][i][j] - m.affine_W[key][i][j]); v.affine_W[key][i][j] += (1.0f - beta2) * (grads.affine_W[key][i][j] * grads.affine_W[key][i][j] - v.affine_W[key][i][j]); params.affine_W[key][i][j] -= lr_t * m.affine_W[key][i][j] / (sqrt(v.affine_W[key][i][j]) + 1e-7f); } }
	for (auto& pair : params.b) { string key = pair.first; for (size_t i = 0; i < params.b[key].size(); ++i) { m.b[key][i] += (1.0f - beta1) * (grads.b[key][i] - m.b[key][i]); v.b[key][i] += (1.0f - beta2) * (grads.b[key][i] * grads.b[key][i] - v.b[key][i]); params.b[key][i] -= lr_t * m.b[key][i] / (sqrt(v.b[key][i]) + 1e-7f); } }
}

// ---------- mnist.cpp ----------
const string url_base = "http://yann.lecun.com/exdb/mnist/";
const string dataset_dir = "dataset/";
const vector<string> key_files = { dataset_dir + "train-images-idx3-ubyte", dataset_dir + "train-labels-idx1-ubyte", dataset_dir + "t10k-images-idx3-ubyte", dataset_dir + "t10k-labels-idx1-ubyte" };
const int img_size = 784;
uint32_t swap_endian(uint32_t val) { return ((val >> 24) & 0xff) | ((val << 8) & 0xff0000) | ((val >> 8) & 0xff00) | ((val << 24) & 0xff000000); }
void _download(const string& file_name) {
	string gz_file = file_name + ".gz";
	ifstream f(file_name, ios::binary); if (f.good()) { f.close(); return; }
	cout << "Downloading " << gz_file << " ... " << endl;
	string download_cmd = "curl -O " + url_base + gz_file;
	if (system(download_cmd.c_str()) != 0) { cerr << "[에러] 다운로드 실패. 시스템에 curl이 설치되어 있는지 확인하세요." << endl; exit(1); }
	cout << "Done" << endl;
	cout << "Extracting " << gz_file << " ... " << endl; string unzip_cmd = "gzip -d " + gz_file; system(unzip_cmd.c_str()); cout << "Done" << endl;
}
void download_mnist() { for (const string& file : key_files) _download(file); }
Matrix _change_one_hot_label(const vector<int>& X) { Matrix T(X.size(), vector<float>(10, 0.0f)); for (size_t i=0;i<X.size();++i) T[i][X[i]] = 1.0f; return T; }
MNIST_Dataset load_mnist(bool normalize, bool flatten, bool one_hot_label) {
	download_mnist(); MNIST_Dataset dataset;
	for (int idx = 0; idx < 2; ++idx) {
		string file_name = (idx == 0) ? key_files[0] : key_files[2]; ifstream file(file_name, ios::binary);
		cout << "Converting " << file_name << " to C++ Vector ..." << endl;
		uint32_t magic = 0, num_images = 0, rows = 0, cols = 0; file.read((char*)&magic, 4); file.read((char*)&num_images, 4); file.read((char*)&rows, 4); file.read((char*)&cols, 4);
		num_images = swap_endian(num_images); rows = swap_endian(rows); cols = swap_endian(cols);
		if (flatten) { Matrix data_2d(num_images, vector<float>(img_size, 0.0f)); for (uint32_t i=0;i<num_images;++i) for (int p=0;p<img_size;++p){ unsigned char pixel=0; file.read((char*)&pixel,1); data_2d[i][p] = normalize ? ((float)pixel/255.0f) : (float)pixel; } if (idx==0) dataset.train_img_2d = move(data_2d); else dataset.test_img_2d = move(data_2d); }
		else { Tensor data_4d(num_images, vector<vector<vector<float>>>(1, vector<vector<float>>(rows, vector<float>(cols,0.0f)))); for (uint32_t i=0;i<num_images;++i) for (uint32_t r=0;r<rows;++r) for (uint32_t c=0;c<cols;++c){ unsigned char pixel=0; file.read((char*)&pixel,1); data_4d[i][0][r][c] = normalize ? ((float)pixel/255.0f) : (float)pixel; } if (idx==0) dataset.train_img_4d = move(data_4d); else dataset.test_img_4d = move(data_4d); }
		file.close();
	}
	for (int idx=0; idx<2; ++idx) {
		string file_name = (idx == 0) ? key_files[1] : key_files[3]; ifstream file(file_name, ios::binary);
		cout << "Converting " << file_name << " to C++ Vector ..." << endl;
		uint32_t magic = 0, num_items = 0; file.read((char*)&magic,4); file.read((char*)&num_items,4); num_items = swap_endian(num_items);
		vector<int> labels(num_items); for (uint32_t i=0;i<num_items;++i){ unsigned char label=0; file.read((char*)&label,1); labels[i] = (int)label; } file.close();
		if (one_hot_label) { if (idx==0) dataset.train_label_one_hot = _change_one_hot_label(labels); else dataset.test_label_one_hot = _change_one_hot_label(labels); }
		else { if (idx==0) dataset.train_label = move(labels); else dataset.test_label = move(labels); }
	}
	cout << "Done!" << endl; return dataset;
}

// ---------- deep_convnet.cpp ----------
static Tensor mat2tensor(const Matrix& m) { int N=m.size(), D=m[0].size(); Tensor t(N, vector<vector<vector<float>>>(D, vector<vector<float>>(1, vector<float>(1)))); for (int i=0;i<N;++i) for (int j=0;j<D;++j) t[i][j][0][0] = m[i][j]; return t; }
static Matrix tensor2mat(const Tensor& t) { int N=t.size(), D=t[0].size(); Matrix m(N, vector<float>(D)); for (int i=0;i<N;++i) for (int j=0;j<D;++j) m[i][j] = t[i][j][0][0]; return m; }
static Tensor randn_tensor(int FN, int C, int FH, int FW, float scale) { Tensor t(FN, vector<vector<vector<float>>>(C, vector<vector<float>>(FH, vector<float>(FW)))); random_device rd; mt19937 gen(rd()); normal_distribution<float> dist(0.0f,1.0f); for (int fn=0;fn<FN;++fn) for (int c=0;c<C;++c) for (int fh=0;fh<FH;++fh) for (int fw=0;fw<FW;++fw) t[fn][c][fh][fw] = dist(gen) * scale; return t; }
static Matrix randn_matrix(int row, int col, float scale) { Matrix m(row, vector<float>(col)); random_device rd; mt19937 gen(rd()); normal_distribution<float> dist(0.0f,1.0f); for (int r=0;r<row;++r) for (int c=0;c<col;++c) m[r][c] = dist(gen) * scale; return m; }

DeepConvNet::DeepConvNet() {
	int hidden_size = 50; int output_size = 10; vector<int> pre_node_nums = { 1 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 4 * 4, hidden_size };
	vector<float> weight_scales; for (int n : pre_node_nums) weight_scales.push_back(sqrt(2.0f / n));
	params.conv_W["W1"] = randn_tensor(16,1,3,3,weight_scales[0]); params.b["b1"] = vector<float>(16,0.0f);
	params.conv_W["W2"] = randn_tensor(16,16,3,3,weight_scales[1]); params.b["b2"] = vector<float>(16,0.0f);
	params.conv_W["W3"] = randn_tensor(32,16,3,3,weight_scales[2]); params.b["b3"] = vector<float>(32,0.0f);
	params.conv_W["W4"] = randn_tensor(32,32,3,3,weight_scales[3]); params.b["b4"] = vector<float>(32,0.0f);
	params.conv_W["W5"] = randn_tensor(64,32,3,3,weight_scales[4]); params.b["b5"] = vector<float>(64,0.0f);
	params.conv_W["W6"] = randn_tensor(64,64,3,3,weight_scales[5]); params.b["b6"] = vector<float>(64,0.0f);
	params.affine_W["W7"] = randn_matrix(64 * 4 * 4, hidden_size, weight_scales[6]); params.b["b7"] = vector<float>(hidden_size,0.0f);
	params.affine_W["W8"] = randn_matrix(hidden_size, output_size, weight_scales[7]); params.b["b8"] = vector<float>(output_size,0.0f);
	conv1 = new Convolution(params.conv_W["W1"], params.b["b1"], 1, 1); relu1 = new Relu();
	conv2 = new Convolution(params.conv_W["W2"], params.b["b2"], 1, 1); relu2 = new Relu(); pool1 = new Pooling(2,2,2,0);
	conv3 = new Convolution(params.conv_W["W3"], params.b["b3"], 1, 1); relu3 = new Relu();
	conv4 = new Convolution(params.conv_W["W4"], params.b["b4"], 1, 2); relu4 = new Relu(); pool2 = new Pooling(2,2,2,0);
	conv5 = new Convolution(params.conv_W["W5"], params.b["b5"], 1, 1); relu5 = new Relu();
	conv6 = new Convolution(params.conv_W["W6"], params.b["b6"], 1, 1); relu6 = new Relu(); pool3 = new Pooling(2,2,2,0);
	affine1 = new Affine(params.affine_W["W7"], params.b["b7"]); relu7 = new Relu(); drop1 = new Dropout(0.5f);
	affine2 = new Affine(params.affine_W["W8"], params.b["b8"]); drop2 = new Dropout(0.5f);
	last_layer = new SoftmaxLoss();
}

DeepConvNet::~DeepConvNet() {
	delete conv1; delete relu1; delete conv2; delete relu2; delete pool1;
	delete conv3; delete relu3; delete conv4; delete relu4; delete pool2;
	delete conv5; delete relu5; delete conv6; delete relu6; delete pool3;
	delete affine1; delete relu7; delete drop1; delete affine2; delete drop2; delete last_layer;
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

float DeepConvNet::loss(const Tensor& x, const vector<int>& t) { Matrix y = predict(x, true); return last_layer->forward(y, t); }

float DeepConvNet::accuracy(const Tensor& x, const vector<int>& t, int batch_size) {
	float acc = 0.0f; int N = x.size(); for (int i=0;i<N / batch_size;i++) { Tensor tx(x.begin() + i * batch_size, x.begin() + (i + 1) * batch_size); vector<int> tt(t.begin() + i * batch_size, t.begin() + (i + 1) * batch_size); Matrix y = predict(tx, false); for (int j=0;j<batch_size;++j) { int max_idx=0; float max_val = y[j][0]; for (size_t c=1;c<y[j].size();++c) if (y[j][c] > max_val) { max_val = y[j][c]; max_idx = c; } if (max_idx == tt[j]) acc += 1.0f; } } return acc / N; }

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
	auto write_string = [&](const string& s) { size_t len = s.length(); ofs.write((char*)&len, sizeof(size_t)); ofs.write(s.c_str(), len); };
	size_t conv_size = params.conv_W.size(); ofs.write((char*)&conv_size, sizeof(size_t));
	for (auto& pair : params.conv_W) { write_string(pair.first); int N = pair.second.size(), C = pair.second[0].size(), H = pair.second[0][0].size(), W = pair.second[0][0][0].size(); ofs.write((char*)&N, sizeof(int)); ofs.write((char*)&C, sizeof(int)); ofs.write((char*)&H, sizeof(int)); ofs.write((char*)&W, sizeof(int)); for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int h=0;h<H;++h) ofs.write((char*)pair.second[n][c][h].data(), W * sizeof(float)); }
	size_t affine_size = params.affine_W.size(); ofs.write((char*)&affine_size, sizeof(size_t)); for (auto& pair : params.affine_W) { write_string(pair.first); int R = pair.second.size(), C = pair.second[0].size(); ofs.write((char*)&R, sizeof(int)); ofs.write((char*)&C, sizeof(int)); for (int r=0;r<R;++r) ofs.write((char*)pair.second[r].data(), C * sizeof(float)); }
	size_t b_size = params.b.size(); ofs.write((char*)&b_size, sizeof(size_t)); for (auto& pair : params.b) { write_string(pair.first); int S = pair.second.size(); ofs.write((char*)&S, sizeof(int)); ofs.write((char*)pair.second.data(), S * sizeof(float)); }
	ofs.close();
}

void DeepConvNet::load_params(const string& file_name) {
	ifstream ifs(file_name, ios::binary); if (!ifs.is_open()) return;
	auto read_string = [&]() { size_t len; ifs.read((char*)&len, sizeof(size_t)); string s(len, ' '); ifs.read(&s[0], len); return s; };
	size_t conv_size; ifs.read((char*)&conv_size, sizeof(size_t)); for (size_t i=0;i<conv_size;++i) { string key = read_string(); int N,C,H,W; ifs.read((char*)&N,sizeof(int)); ifs.read((char*)&C,sizeof(int)); ifs.read((char*)&H,sizeof(int)); ifs.read((char*)&W,sizeof(int)); Tensor t(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(W)))); for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int h=0;h<H;++h) ifs.read((char*)t[n][c][h].data(), W * sizeof(float)); params.conv_W[key] = t; }
	size_t affine_size; ifs.read((char*)&affine_size, sizeof(size_t)); for (size_t i=0;i<affine_size;++i) { string key = read_string(); int R,C; ifs.read((char*)&R,sizeof(int)); ifs.read((char*)&C,sizeof(int)); Matrix m(R, vector<float>(C)); for (int r=0;r<R;++r) ifs.read((char*)m[r].data(), C * sizeof(float)); params.affine_W[key] = m; }
	size_t b_size; ifs.read((char*)&b_size, sizeof(size_t)); for (size_t i=0;i<b_size;++i) { string key = read_string(); int S; ifs.read((char*)&S,sizeof(int)); vector<float> v(S); ifs.read((char*)v.data(), S * sizeof(float)); params.b[key] = v; }
	ifs.close(); sync_params_to_layers();
}

void DeepConvNet::sync_params_to_layers() {
	conv1->W = params.conv_W["W1"]; conv1->b = params.b["b1"]; conv2->W = params.conv_W["W2"]; conv2->b = params.b["b2"]; conv3->W = params.conv_W["W3"]; conv3->b = params.b["b3"]; conv4->W = params.conv_W["W4"]; conv4->b = params.b["b4"]; conv5->W = params.conv_W["W5"]; conv5->b = params.b["b5"]; conv6->W = params.conv_W["W6"]; conv6->b = params.b["b6"]; affine1->W = params.affine_W["W7"]; affine1->b = params.b["b7"]; affine2->W = params.affine_W["W8"]; affine2->b = params.b["b8"]; }

// ---------- trainer.cpp ----------
Trainer::Trainer(DeepConvNet& network, const Tensor& x_train, const vector<int>& t_train, const Tensor& x_test, const vector<int>& t_test, int epochs, int mini_batch_size, string optimizer, float lr, int evaluate_sample_num_per_epoch, bool verbose)
	: network(network), x_train(x_train), t_train(t_train), x_test(x_test), t_test(t_test)
{
	this->epochs = epochs; this->batch_size = mini_batch_size; this->evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch; this->verbose = verbose;
	this->optimizer_name = optimizer; for (char& c : this->optimizer_name) c = tolower(c);
	sgd = SGD(lr); momentum = Momentum(lr); nesterov = Nesterov(lr); adagrad = AdaGrad(lr); rmsprop = RMSprop(lr); adam = Adam(lr);
	train_size = x_train.size(); iter_per_epoch = max(train_size / mini_batch_size, 1); max_iter = epochs * iter_per_epoch; current_iter = 0; current_epoch = 0;
}

void Trainer::train_step() {
	Tensor x_batch(batch_size); vector<int> t_batch(batch_size);
	random_device rd; mt19937 gen(rd()); uniform_int_distribution<int> dist(0, train_size - 1);
	for (int i=0;i<batch_size;++i) { int idx = dist(gen); x_batch[i] = x_train[idx]; t_batch[i] = t_train[idx]; }
	ParamsDict grads = network.gradient(x_batch, t_batch);
	if (optimizer_name == "sgd") sgd.update(network.params, grads);
	else if (optimizer_name == "momentum") momentum.update(network.params, grads);
	else if (optimizer_name == "nesterov") nesterov.update(network.params, grads);
	else if (optimizer_name == "adagrad") adagrad.update(network.params, grads);
	else if (optimizer_name == "rmsprop") rmsprop.update(network.params, grads);
	else if (optimizer_name == "adam") adam.update(network.params, grads);
	network.sync_params_to_layers();
	float loss = network.loss(x_batch, t_batch); train_loss_list.push_back(loss);
	if (verbose) cout << "train loss: " << loss << "\n";
	if (current_iter % iter_per_epoch == 0) {
		current_epoch++;
		Tensor x_train_sample, x_test_sample; vector<int> t_train_sample, t_test_sample;
		if (evaluate_sample_num_per_epoch > 0) {
			int t_len = evaluate_sample_num_per_epoch;
			x_train_sample.assign(x_train.begin(), x_train.begin() + min(t_len, (int)x_train.size()));
			t_train_sample.assign(t_train.begin(), t_train.begin() + min(t_len, (int)t_train.size()));
			x_test_sample.assign(x_test.begin(), x_test.begin() + min(t_len, (int)x_test.size()));
			t_test_sample.assign(t_test.begin(), t_test.begin() + min(t_len, (int)t_test.size()));
		} else { x_train_sample = x_train; t_train_sample = t_train; x_test_sample = x_test; t_test_sample = t_test; }
		float train_acc = network.accuracy(x_train_sample, t_train_sample); float test_acc = network.accuracy(x_test_sample, t_test_sample);
		train_acc_list.push_back(train_acc); test_acc_list.push_back(test_acc);
		if (verbose) cout << "=== epoch: " << current_epoch << ", train acc: " << train_acc << ", test acc: " << test_acc << " ===\n";
	}
	current_iter++;
}

void Trainer::train() { for (int i=0;i<max_iter;++i) train_step(); float test_acc = network.accuracy(x_test, t_test); if (verbose) { cout << "=============== Final Test Accuracy ===============\n"; cout << "test acc: " << test_acc << "\n"; } }
