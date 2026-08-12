// 파일 위치: PytoCpp_CNN/layers.cpp
// 파일 명: layers.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "util.h"
#include "layers.h"

using namespace std;

// ==========================================
// Relu
// ==========================================
Relu::Relu() {}

Tensor Relu::forward(const Tensor& x) {
    int N = x.size();
    int C = x[0].size();
    int H = x[0][0].size();
    int W = x[0][0][0].size();

    mask = Tensor(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(W, 0.0f))));
    Tensor out = x;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    if (x[n][c][h][w] <= 0) {
                        mask[n][c][h][w] = 1.0f;
                        out[n][c][h][w] = 0.0f;
                    }
                }
            }
        }
    }
    return out;
}

Tensor Relu::backward(const Tensor& dout) {
    int N = dout.size();
    int C = dout[0].size();
    int H = dout[0][0].size();
    int W = dout[0][0][0].size();

    Tensor dx = dout;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    if (mask[n][c][h][w] == 1.0f) {
                        dx[n][c][h][w] = 0.0f;
                    }
                }
            }
        }
    }

    return dx;
}

// ==========================================
// Sigmoid
// ==========================================
Sigmoid::Sigmoid() {}

Tensor Sigmoid::forward(const Tensor& x) {
    int N = x.size();
    int C = x[0].size();
    int H = x[0][0].size();
    int W = x[0][0][0].size();

    Tensor out = x;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    out[n][c][h][w] = 1.0f / (1.0f + exp(-x[n][c][h][w]));
                }
            }
        }
    }
    out_val = out;

    return out;
}

Tensor Sigmoid::backward(const Tensor& dout) {
    int N = dout.size();
    int C = dout[0].size();
    int H = dout[0][0].size();
    int W = dout[0][0][0].size();

    Tensor dx = dout;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    dx[n][c][h][w] = dout[n][c][h][w] * (1.0f - out_val[n][c][h][w]);
                }
            }
        }
    }
    return dx;
}

// ==========================================
// Affine
// ==========================================
Affine::Affine(const Matrix& W_init, const vector<float>& b_init) {
    this->W = W_init;
    this->b = b_init;
}

Matrix Affine::forward(const Tensor& x_in) {
    int N = x_in.size();
    int C = x_in[0].size();
    int H = x_in[0][0].size();
    int W_dim = x_in[0][0][0].size();

    original_x_shape = { N,C,H,W_dim };

    int D = C * H * W_dim;
    x_mat = Matrix(N, vector<float>(D, 0.0f));

    for (int n = 0; n < N; n++) {
        int idx = 0;
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W_dim; w++) {
                    x_mat[n][idx++] = x_in[n][c][h][w];
                }
            }
        }
    }

    Matrix out(N, vector<float>(W[0].size(), 0.0f));

    for (int n = 0; n < N; n++) {
        for (size_t j = 0; j < W[0].size(); j++) {
            for (size_t k = 0; k < x_mat[0].size(); k++) {
                out[n][j] += x_mat[n][k] * W[k][j];
            }
            out[n][j] += b[j];
        }
    }

    return out;
}

Tensor Affine::backward(const Matrix& dout) {
    int N = dout.size();
    int D = W.size();
    int out_size = W[0].size();

    Matrix dx_mat(N, vector<float>(D, 0.0f));

    for (int n = 0; n < N; n++) {
        for (int j = 0; j < D; j++) {
            for (int k = 0; k < out_size; k++) {
                dx_mat[n][j] += dout[n][k] * W[j][k];
            }
        }
    }

    dW = Matrix(D, vector<float>(out_size, 0.0f));

    for (int i = 0; i < D; i++) {
        for (int j = 0; j < out_size; j++) {
            for (int k = 0; k < N; k++) {
                dW[i][j] += x_mat[k][i] * dout[k][j];
            }
        }
    }

    db = vector<float>(out_size, 0.0f);
    for (int d = 0; d < out_size; d++) {
        for (int n = 0; n < N; n++) {
            db[d] += dout[n][d];
        }
    }

    Tensor dx(original_x_shape[0], vector<vector<vector<float>>>(original_x_shape[1], vector<vector<float>>(original_x_shape[2], vector<float>(original_x_shape[3]))));
    for (int n = 0; n < N; n++) {
        int idx = 0;
        for (int c = 0; c < original_x_shape[1]; c++) {
            for (int h = 0; h < original_x_shape[2]; h++) {
                for (int w = 0; w < original_x_shape[3]; w++) {
                    dx[n][c][h][w] = dx_mat[n][idx++];
                }
            }
        }
    }
    return dx;
}

// ==========================================
// SoftmaxLoss
// ==========================================
SoftmaxLoss::SoftmaxLoss() {}

float SoftmaxLoss::forward(const Matrix& x, const vector<int>& t_in) {
    t = t_in;
    int N = x.size();
    int C = x[0].size();
    y = Matrix(N, vector<float>(C, 0.0f));

    for (int n = 0; n < N; n++) {
        float max_val = x[n][0];
        for (int c = 1; c < C; c++) {
            if (x[n][c] > max_val) {
                max_val = x[n][c];
            }
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < C; c++) {
            y[n][c] = exp(x[n][c] - max_val);
            sum_exp += y[n][c];
        }
        for (int c = 0; c < C; c++) {
            y[n][c] /= sum_exp;
        }
    }

    loss_val = 0.0f;
    for (int n = 0; n < N; n++) {
        loss_val -= log(y[n][t[n]] + 1e-7f);
    }
    loss_val /= N;

    return loss_val;
}

Matrix SoftmaxLoss::backward(float dout) {
    int batch_size = t.size();
    int C = y[0].size();
    Matrix dx = y;

    for (int n = 0; n < batch_size; n++) {
        dx[n][t[n]] -= 1.0f;
        for (int c = 0; c < C; c++) {
            dx[n][c] = (dx[n][c] * dout) / batch_size;
        }
    }

    return dx;
}

// ==========================================
// Dropout
// ==========================================
Dropout::Dropout(float ratio) {
    dropout_ratio = ratio;
}

Tensor Dropout::forward(const Tensor& x, bool train_flg) {
    int N = x.size(), C = x[0].size(), H = x[0][0].size(), W = x[0][0][0].size();
    Tensor out = x;

    if (train_flg) {
        mask = Tensor(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(W, 0.0f))));
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        float rand_val = static_cast<float>(rand()) / RAND_MAX;
                        if (rand_val > dropout_ratio) {
                            mask[n][c][h][w] = 1.0f;
                        }
                        else {
                            mask[n][c][h][w] = 0.0f;
                            out[n][c][h][w] = 0.0f;
                        }
                    }
                }
            }
        }
        return out;
    }
    else {
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        out[n][c][h][w] = x[n][c][h][w] * (1.0f - dropout_ratio);
                    }
                }
            }
        }
        return out;
    }
}

Tensor Dropout::backward(const Tensor& dout) {
    int N = dout.size(), C = dout[0].size(), H = dout[0][0].size(), W = dout[0][0][0].size();
    Tensor dx = dout;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    dx[n][c][h][w] = dout[n][c][h][w] * mask[n][c][h][w];
                }
            }
        }
    }
    return dx;
}

// ==========================================
// Convolution
// ==========================================
Convolution::Convolution(const Tensor& W_init, const vector<float>& b_init, int stride, int pad) {
    this->W = W_init;
    this->b = b_init;
    this->stride = stride;
    this->pad = pad;
}

Tensor Convolution::forward(const Tensor& x_in) {
    int FN = W.size(), C = W[0].size(), FH = W[0][0].size(), FW = W[0][0][0].size();
    int N = x_in.size(), H = x_in[0][0].size(), W_dim = x_in[0][0][0].size();

    int out_h = 1 + (H + 2 * pad - FH) / stride;
    int out_w = 1 + (W_dim + 2 * pad - FW) / stride;

    col = im2col(x_in, FH, FW, stride, pad);

    col_W = Matrix(C * FH * FW, vector<float>(FN, 0.0f));
    for (int fn = 0; fn < FN; ++fn) {
        int idx = 0;
        for (int c = 0; c < C; ++c) {
            for (int fh = 0; fh < FH; ++fh) {
                for (int fw = 0; fw < FW; ++fw) {
                    col_W[idx++][fn] = W[fn][c][fh][fw];
                }
            }
        }
    }

    Matrix out_mat(col.size(), vector<float>(col_W[0].size(), 0.0f));
    for (size_t i = 0; i < col.size(); ++i) {
        for (size_t j = 0; j < col_W[0].size(); ++j) {
            for (size_t k = 0; k < col[0].size(); ++k) {
                out_mat[i][j] += col[i][k] * col_W[k][j];
            }
            out_mat[i][j] += b[j];
        }
    }

    Tensor out(N, vector<vector<vector<float>>>(FN, vector<vector<float>>(out_h, vector<float>(out_w))));
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int row_idx = (n * out_h + oh) * out_w + ow;
                for (int fn = 0; fn < FN; ++fn) {
                    out[n][fn][oh][ow] = out_mat[row_idx][fn];
                }
            }
        }
    }

    x = x_in;
    return out;
}

Tensor Convolution::backward(const Tensor& dout) {
    int FN = W.size(), C = W[0].size(), FH = W[0][0].size(), FW = W[0][0][0].size();
    int N = dout.size(), out_h = dout[0][0].size(), out_w = dout[0][0][0].size();

    Matrix dout_mat(N * out_h * out_w, vector<float>(FN, 0.0f));
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int row_idx = (n * out_h + oh) * out_w + ow;
                for (int fn = 0; fn < FN; ++fn) {
                    dout_mat[row_idx][fn] = dout[n][fn][oh][ow];
                }
            }
        }
    }

    db = vector<float>(FN, 0.0f);
    for (size_t i = 0; i < dout_mat.size(); ++i) {
        for (int fn = 0; fn < FN; ++fn) {
            db[fn] += dout_mat[i][fn];
        }
    }

    Matrix dW_mat(col[0].size(), vector<float>(dout_mat[0].size(), 0.0f));
    for (size_t i = 0; i < col[0].size(); ++i) {
        for (size_t j = 0; j < dout_mat[0].size(); ++j) {
            for (size_t k = 0; k < col.size(); ++k) {
                dW_mat[i][j] += col[k][i] * dout_mat[k][j];
            }
        }
    }

    dW = Tensor(FN, vector<vector<vector<float>>>(C, vector<vector<float>>(FH, vector<float>(FW))));
    for (int fn = 0; fn < FN; ++fn) {
        int idx = 0;
        for (int c = 0; c < C; ++c) {
            for (int fh = 0; fh < FH; ++fh) {
                for (int fw = 0; fw < FW; ++fw) {
                    dW[fn][c][fh][fw] = dW_mat[idx++][fn];
                }
            }
        }
    }

    Matrix dcol(dout_mat.size(), vector<float>(col_W.size(), 0.0f));
    for (size_t i = 0; i < dout_mat.size(); ++i) {
        for (size_t j = 0; j < col_W.size(); ++j) {
            for (size_t k = 0; k < dout_mat[0].size(); ++k) {
                dcol[i][j] += dout_mat[i][k] * col_W[j][k];
            }
        }
    }

    Tensor dx = col2im(dcol, N, x[0].size(), x[0][0].size(), x[0][0][0].size(), FH, FW, stride, pad);
    return dx;
}

//// ==========================================
//// Pooling (4차원 Tensor 직접 순회 버전 - 메모리 안전)
//// ==========================================
//Pooling::Pooling(int pool_h, int pool_w, int stride, int pad) {
//    this->pool_h = pool_h;
//    this->pool_w = pool_w;
//    this->stride = stride;
//    this->pad = pad;
//}
//
//Tensor Pooling::forward(const Tensor& x_in) {
//    int N = x_in.size();
//    int C = x_in[0].size();
//    int H = x_in[0][0].size();
//    int W_dim = x_in[0][0][0].size();
//
//    int out_h = 1 + (H - pool_h) / stride;
//    int out_w = 1 + (W_dim - pool_w) / stride;
//
//    Tensor out(N, vector<vector<vector<float>>>(C, vector<vector<float>>(out_h, vector<float>(out_w, 0.0f))));
//
//    // 역전파 때 최댓값의 위치를 정확히 기억하기 위한 4차원 구조체 배열 초기화
//    arg_max_h = vector<vector<vector<vector<int>>>>(N, vector<vector<vector<int>>>(C, vector<vector<int>>(out_h, vector<int>(out_w, 0))));
//    arg_max_w = vector<vector<vector<vector<int>>>>(N, vector<vector<vector<int>>>(C, vector<vector<int>>(out_h, vector<int>(out_w, 0))));
//
//    for (int n = 0; n < N; ++n) {
//        for (int c = 0; c < C; ++c) {
//            for (int oh = 0; oh < out_h; ++oh) {
//                for (int ow = 0; ow < out_w; ++ow) {
//                    float max_val = -1e9f;
//                    int max_h = 0, max_w = 0;
//
//                    for (int ph = 0; ph < pool_h; ++ph) {
//                        for (int pw = 0; pw < pool_w; ++pw) {
//                            int h_idx = oh * stride + ph;
//                            int w_idx = ow * stride + pw;
//                            float val = x_in[n][c][h_idx][w_idx];
//                            if (val > max_val) {
//                                max_val = val;
//                                max_h = h_idx;
//                                max_w = w_idx;
//                            }
//                        }
//                    }
//                    out[n][c][oh][ow] = max_val;
//                    arg_max_h[n][c][oh][ow] = max_h;
//                    arg_max_w[n][c][oh][ow] = max_w;
//                }
//            }
//        }
//    }
//
//    x = x_in;
//    return out;
//}
//
//Tensor Pooling::backward(const Tensor& dout) {
//    int N = x.size();
//    int C = x[0].size();
//    int H = x[0][0].size();
//    int W_dim = x[0][0][0].size();
//    int out_h = dout[0][0].size();
//    int out_w = dout[0][0][0].size();
//
//    Tensor dx(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(W_dim, 0.0f))));
//
//    for (int n = 0; n < N; ++n) {
//        for (int c = 0; c < C; ++c) {
//            for (int oh = 0; oh < out_h; ++oh) {
//                for (int ow = 0; ow < out_w; ++ow) {
//                    int h_idx = arg_max_h[n][c][oh][ow];
//                    int w_idx = arg_max_w[n][c][oh][ow];
//                    dx[n][c][h_idx][w_idx] += dout[n][c][oh][ow];
//                }
//            }
//        }
//    }
//
//    return dx;
//}

// ==========================================
// Pooling (책의 원본 im2col 및 전개/성형 구조 100% 반영)
// ==========================================
Pooling::Pooling(int pool_h, int pool_w, int stride, int pad) {
    this->pool_h = pool_h;
    this->pool_w = pool_w;
    this->stride = stride;
    this->pad = pad;
}

Tensor Pooling::forward(const Tensor& x_in) {
    int N = x_in.size();
    int C = x_in[0].size();
    int H = x_in[0][0].size();
    int W_dim = x_in[0][0][0].size();

    int out_h = 1 + (H - pool_h) / stride;
    int out_w = 1 + (W_dim - pool_w) / stride;

    // 1. 책의 풀링 im2col 방식: 채널별로 독립적인 윈도우 펼치기
    // 행 크기: N * C * out_h * out_w, 열 크기: pool_h * pool_w
    int total_rows = N * C * out_h * out_w;
    int pool_size = pool_h * pool_w;
    Matrix col(total_rows, vector<float>(pool_size, 0.0f));

    int row_idx = 0;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int idx = 0;
                    for (int ph = 0; ph < pool_h; ++ph) {
                        for (int pw = 0; pw < pool_w; ++pw) {
                            int h_idx = oh * stride + ph;
                            int w_idx = ow * stride + pw;
                            col[row_idx][idx++] = x_in[n][c][h_idx][w_idx];
                        }
                    }
                    row_idx++;
                }
            }
        }
    }

    // 2. argmax 및 최댓값(max) 추출 (책의 np.argmax, np.max 대응)
    arg_max = vector<int>(total_rows);
    vector<float> out_flat(total_rows);

    for (int i = 0; i < total_rows; ++i) {
        float max_val = col[i][0];
        int max_idx = 0;
        for (int j = 1; j < pool_size; ++j) {
            if (col[i][j] > max_val) {
                max_val = col[i][j];
                max_idx = j;
            }
        }
        out_flat[i] = max_val;
        arg_max[i] = max_idx;
    }

    // 3. 성형 (책의 reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) 대응)
    Tensor out(N, vector<vector<vector<float>>>(C, vector<vector<float>>(out_h, vector<float>(out_w))));
    row_idx = 0;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    out[n][c][oh][ow] = out_flat[row_idx++];
                }
            }
        }
    }

    x = x_in;
    return out;
}

Tensor Pooling::backward(const Tensor& dout) {
    int N = dout.size();
    int C = dout[0].size();
    int out_h = dout[0][0].size();
    int out_w = dout[0][0][0].size();
    int pool_size = pool_h * pool_w;
    int total_rows = N * C * out_h * out_w;

    // 책의 dmax 설정 방식 대응
    Matrix dmax(total_rows, vector<float>(pool_size, 0.0f));

    int row_idx = 0;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    dmax[row_idx][arg_max[row_idx]] = dout[n][c][oh][ow];
                    row_idx++;
                }
            }
        }
    }

    // dmax를 원래 입력 크기(dx)로 복원 (col2im의 풀링 전용 버전)
    Tensor dx(N, vector<vector<vector<float>>>(C, vector<vector<float>>(x[0][0].size(), vector<float>(x[0][0][0].size(), 0.0f))));

    row_idx = 0;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int idx = 0;
                    for (int ph = 0; ph < pool_h; ++ph) {
                        for (int pw = 0; pw < pool_w; ++pw) {
                            int h_idx = oh * stride + ph;
                            int w_idx = ow * stride + pw;
                            dx[n][c][h_idx][w_idx] += dmax[row_idx][idx++];
                        }
                    }
                    row_idx++;
                }
            }
        }
    }

    return dx;
}