# 🧠 PytoCpp-CNN: 프레임워크 없이 C++로 구현하는 DeepConvNet

![C++](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)
![No Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

『밑바닥부터 시작하는 딥러닝 1』의 Python 기반 고정밀 합성곱 신경망(DeepConvNet)을 **외부 선형대수 라이브러리(NumPy 등) 없이 오직 C++ 표준 라이브러리(`std::vector`)만을 사용하여 밑바닥부터 재구현(Converting)**한 프로젝트입니다.<br>
 Reference: 『밑바닥부터 시작하는 딥러닝 1』 (사이토 고키 저, 한빛미디어)
## 📖 프로젝트 기획 의도

이 프로젝트의 목적은 '극한의 연산 최적화'가 아닙니다. 
블랙박스처럼 동작하는 딥러닝 프레임워크와 NumPy의 마법을 걷어내고, **데이터가 메모리 위에서 어떻게 흐르고, 다차원 배열의 인덱싱이 어떻게 꼬이며, 역전파 기울기가 어떻게 전달되는지 컴퓨터 구조적 관점에서 온전히 이해하기 위해** 시작되었습니다. 

### 📝 기술 블로그 포스팅 (상세 구현 과정 및 트러블슈팅)
* [✅ [Practice] 파이썬 코드를 C++로 컨버팅하며 배우는 CNN 밑바닥 구현](https://blog.naver.com/hk100456/224376919938)

---

## ✨ 핵심 구현 특징

블로그에 작성된 실제 구현 경험을 바탕으로 한 핵심 기능들입니다.

* **NumPy 다차원 배열의 4차원 텐서(Tensor)화:** 파이썬의 직관적인 인덱싱(`x[n][c][h][w]`)을 모사하기 위해 `vector<vector<vector<vector<float>>>>` 형태로 다차원 배열을 직접 정의하여 사용했습니다.
* **수제 `im2col` & `col2im`:** 파이썬 슬라이싱으로 처리되던 로직을 자비 없는 6중 `for` 문으로 구현하며, 4차원 데이터가 2차원 행렬로 어떻게 전개되는지 뼈대부터 조립했습니다.
* **채널 독립적 Pooling (버그 해결):** 합성곱(Conv)과 달리 풀링은 채널간 값을 절대 섞지 않고 독립적으로 연산해야 함을 깨닫고, 4차원 텐서 형태를 유지한 채 역전파용 위치(`arg_max`)를 기억하는 정석적인 Max Pooling 로직을 구현했습니다.
* **객체지향 레이어(Layer) 설계:** 각 계층(`Convolution`, `Pooling`, `Affine`, `ReLU`, `Dropout`, `SoftmaxLoss`)을 공통된 `forward`와 `backward` 메서드를 갖는 독립된 모듈 클래스로 캡슐화하여, 이들을 레고 블록처럼 조립해 네트워크를 구성했습니다.

---

## 🏗️ 네트워크 아키텍처 (Network Architecture)

총 6개의 합성곱 층과 2개의 완전연결 층으로 구성된 신경망입니다.

1. `Conv` -> `ReLU`
2. `Conv` -> `ReLU` -> `Pooling`
3. `Conv` -> `ReLU`
4. `Conv` -> `ReLU` -> `Pooling`
5. `Conv` -> `ReLU`
6. `Conv` -> `ReLU` -> `Pooling`
7. `Affine` -> `ReLU` -> `Dropout`
8. `Affine` -> `Dropout` -> `SoftmaxLoss`

---

## ⏱️ 학습 성능 및 회고 (Lessons Learned)

* **테스트 환경:** LG Gram (순수 CPU 연산)
* **소요 시간:** 1에폭(Epoch) 학습에 약 **4시간** 소요 (Release /O2 최적화 적용 기준)
* **정확도:** 1에폭 완료 후 손실(Loss) 1.0 부근, 테스트 정확도 **0.97** 달성

파이썬 코드의 직관적인 구조를 흉내 내기 위해 중첩 `vector`를 사용한 탓에 극심한 메모리 오버헤드와 캐시 미스(Cache Miss)가 발생했습니다. 이 비효율적인 4시간의 벤치마크를 통해, 파이썬이 내부적으로 C/C++ 기반의 고속 행렬 연산 라이브러리(NumPy)를 사용하고 딥러닝 연산에서 **하드웨어 가속(GPU 병렬 처리) 및 메모리 구조 최적화가 왜 필수적인지** 몸소 체험할 수 있었습니다.

---

## 🚀 빌드 및 실행 방법 (Build & Run)

외부 의존성이 없으므로 C++11 이상을 지원하는 컴파일러에서 바로 빌드할 수 있습니다. 
> ⚠️ **주의:** 순수 CPU 기반의 무거운 `for` 문 연산이 진행되므로 컴파일러 최적화 옵션(`-O2` 또는 `-O3`)을 반드시 활성화해야 합니다.

```bash
# 1. 저장소 클론
$ git clone [https://github.com/hakyaung/PytoCpp-CNN.git](https://github.com/hakyaung/PytoCpp-CNN.git)
$ cd PytoCpp-CNN

# 2. 최적화 플래그를 적용하여 빌드 (Linux/Mac 기준)
$ g++ -O2 -std=c++11 src/*.cpp -o train_deepnet

# 3. 학습 실행
$ ./train_deepnet
