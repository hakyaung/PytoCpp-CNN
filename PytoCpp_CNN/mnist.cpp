// 파일 위치: src/dataset/mnist.cpp
// 파일 명: mnist.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdint>
#include "mnist.h"

using namespace std;

const string url_base = "http://yann.lecun.com/exdb/mnist/";
const string dataset_dir = "dataset/";
const vector<string> key_files = {
    dataset_dir + "train-images.idx3-ubyte", // 하이픈을 점으로 변경
    dataset_dir + "train-labels.idx1-ubyte",
    dataset_dir + "t10k-images.idx3-ubyte",
    dataset_dir + "t10k-labels.idx1-ubyte"
};

const int img_size = 784;

// MNIST 데이터는 Big-Endian으로 저장되어 있어 바이트 순서를 뒤집어주어야 합니다.
uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff) |
        ((val << 8) & 0xff0000) |
        ((val >> 8) & 0xff00) |
        ((val << 24) & 0xff000000);
}

// _download 함수 포팅
void _download(const string& file_name) {
    string gz_file = file_name + ".gz";

    // 이미 압축 해제된 파일이 있으면 패스
    ifstream f(file_name, ios::binary);
    if (f.good()) {
        f.close();
        return;
    }

    cout << "Downloading " << gz_file << " ... " << endl;
    // C++ 표준에는 HTTP 다운로드가 없어 시스템 명령어로 curl 호출
    string download_cmd = "curl -O " + url_base + gz_file;
    if (system(download_cmd.c_str()) != 0) {
        cerr << "[에러] 다운로드 실패. 시스템에 curl이 설치되어 있는지 확인하세요." << endl;
        exit(1);
    }
    cout << "Done" << endl;

    cout << "Extracting " << gz_file << " ... " << endl;
    string unzip_cmd = "gzip -d " + gz_file;
    system(unzip_cmd.c_str());
    cout << "Done" << endl;
}

void download_mnist() {
    for (const string& file : key_files) {
        _download(file);
    }
}

// _change_one_hot_label 함수 포팅
Matrix _change_one_hot_label(const vector<int>& X) {
    Matrix T(X.size(), vector<float>(10, 0.0f));
    for (size_t i = 0; i < X.size(); ++i) {
        T[i][X[i]] = 1.0f;
    }
    return T;
}

// _load_img 및 _load_label을 통합하여 바로 구조체에 담아주는 메인 로더
MNIST_Dataset load_mnist(bool normalize, bool flatten, bool one_hot_label) {
    // 1. 파일 다운로드 및 압축 해제 확인
    download_mnist();

    MNIST_Dataset dataset;

    // 2. 이미지 읽기 (Train & Test)
    for (int idx = 0; idx < 2; ++idx) {
        string file_name = (idx == 0) ? key_files[0] : key_files[2];
        ifstream file(file_name, ios::binary);

        cout << "Converting " << file_name << " to C++ Vector ..." << endl;

        uint32_t magic = 0, num_images = 0, rows = 0, cols = 0;
        file.read((char*)&magic, 4);
        file.read((char*)&num_images, 4);
        file.read((char*)&rows, 4);
        file.read((char*)&cols, 4);

        num_images = swap_endian(num_images);
        rows = swap_endian(rows);
        cols = swap_endian(cols);

        if (flatten) {
            Matrix data_2d(num_images, vector<float>(img_size, 0.0f));
            for (uint32_t i = 0; i < num_images; ++i) {
                for (int p = 0; p < img_size; ++p) {
                    unsigned char pixel = 0;
                    file.read((char*)&pixel, 1);
                    data_2d[i][p] = normalize ? ((float)pixel / 255.0f) : (float)pixel;
                }
            }
            if (idx == 0) dataset.train_img_2d = move(data_2d);
            else dataset.test_img_2d = move(data_2d);
        }
        else {
            Tensor data_4d(num_images, vector<vector<vector<float>>>(1, vector<vector<float>>(rows, vector<float>(cols, 0.0f))));
            for (uint32_t i = 0; i < num_images; ++i) {
                for (uint32_t r = 0; r < rows; ++r) {
                    for (uint32_t c = 0; c < cols; ++c) {
                        unsigned char pixel = 0;
                        file.read((char*)&pixel, 1);
                        data_4d[i][0][r][c] = normalize ? ((float)pixel / 255.0f) : (float)pixel;
                    }
                }
            }
            if (idx == 0) dataset.train_img_4d = move(data_4d);
            else dataset.test_img_4d = move(data_4d);
        }
        file.close();
    }

    // 3. 레이블 읽기 (Train & Test)
    for (int idx = 0; idx < 2; ++idx) {
        string file_name = (idx == 0) ? key_files[1] : key_files[3];
        ifstream file(file_name, ios::binary);

        cout << "Converting " << file_name << " to C++ Vector ..." << endl;

        uint32_t magic = 0, num_items = 0;
        file.read((char*)&magic, 4);
        file.read((char*)&num_items, 4);

        num_items = swap_endian(num_items);

        vector<int> labels(num_items);
        for (uint32_t i = 0; i < num_items; ++i) {
            unsigned char label = 0;
            file.read((char*)&label, 1);
            labels[i] = (int)label;
        }
        file.close();

        if (one_hot_label) {
            if (idx == 0) dataset.train_label_one_hot = _change_one_hot_label(labels);
            else dataset.test_label_one_hot = _change_one_hot_label(labels);
        }
        else {
            if (idx == 0) dataset.train_label = move(labels);
            else dataset.test_label = move(labels);
        }
    }

    cout << "Done!" << endl;
    return dataset;
}