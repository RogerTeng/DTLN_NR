#include "DTLN_NR.h"

#include <climits>
#include <cmath>
#include <cstring>

#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/common.h>

#include "kiss_fftr.h"
#include "model_1.h"
#include "model_2.h"

namespace {

constexpr int kWindowSize = 512;
constexpr int kWindowShift = 128;
constexpr int kFftForTensorSize = (kWindowSize / 2 + 1);
constexpr int kNumModels = 2;
constexpr int kNumThreads = 1;

}  // namespace

class DTLN_NR::Impl {
 public:
  int Init();
  void Release();

  int Process(short* input_buffer, short* output_buffer);
  void Denoise();

  TfLiteModel* tflite_model_[kNumModels] = {nullptr, nullptr};
  TfLiteInterpreter* interpreter_[kNumModels] = {nullptr, nullptr};
  TfLiteInterpreterOptions* interpreter_options_ = nullptr;

  TfLiteTensor* input_tensor_[kNumModels][2] = {{nullptr, nullptr},
                                                 {nullptr, nullptr}};
  const TfLiteTensor* output_tensor_[kNumModels][2] = {{nullptr, nullptr},
                                                        {nullptr, nullptr}};

  kiss_fftr_cfg fftr_cfg_ = nullptr;
  kiss_fftr_cfg iffter_cfg_ = nullptr;

  kiss_fft_cpx* input_cpx_ = nullptr;
  kiss_fft_cpx* output_cpx_ = nullptr;

  float* input_buffer_ = nullptr;
  float* output_buffer_ = nullptr;

  float* dtln_freq_output_ = nullptr;
  float* dtln_time_output_ = nullptr;

  int state_size_[kNumModels] = {0, 0};
  float* states_[kNumModels] = {nullptr, nullptr};

  float* input_mag_ = nullptr;
  float* input_phase_ = nullptr;
  float* estimated_block_ = nullptr;

  float* input_sample_ = nullptr;
  float* output_sample_ = nullptr;

  bool init_success_ = false;
};

int DTLN_NR::Impl::Init() {
  int ret = -1;

  do {
    tflite_model_[0] = TfLiteModelCreate(k_lpszModel1Tflite, k_nModel1TfliteLen);
    tflite_model_[1] = TfLiteModelCreate(k_lpszModel2Tflite, k_nModel2TfliteLen);
    if (tflite_model_[0] == nullptr || tflite_model_[1] == nullptr) {
      break;
    }

    interpreter_options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(interpreter_options_, kNumThreads);

    interpreter_[0] = TfLiteInterpreterCreate(tflite_model_[0], interpreter_options_);
    interpreter_[1] = TfLiteInterpreterCreate(tflite_model_[1], interpreter_options_);
    if (interpreter_[0] == nullptr || interpreter_[1] == nullptr) {
      break;
    }

    if (TfLiteInterpreterAllocateTensors(interpreter_[0]) != kTfLiteOk ||
        TfLiteInterpreterAllocateTensors(interpreter_[1]) != kTfLiteOk) {
      break;
    }

    for (int i = 0; i < kNumModels; ++i) {
      input_tensor_[i][0] = TfLiteInterpreterGetInputTensor(interpreter_[i], 0);
      input_tensor_[i][1] = TfLiteInterpreterGetInputTensor(interpreter_[i], 1);

      output_tensor_[i][0] = TfLiteInterpreterGetOutputTensor(interpreter_[i], 0);
      output_tensor_[i][1] = TfLiteInterpreterGetOutputTensor(interpreter_[i], 1);

      state_size_[i] = input_tensor_[i][1]->bytes / sizeof(float);
    }

    fftr_cfg_ = kiss_fftr_alloc(kWindowSize, 0, 0, 0);
    iffter_cfg_ = kiss_fftr_alloc(kWindowSize, 1, 0, 0);

    input_cpx_ = new kiss_fft_cpx[kFftForTensorSize];
    output_cpx_ = new kiss_fft_cpx[kFftForTensorSize];

    input_buffer_ = new float[kWindowSize];
    output_buffer_ = new float[kWindowSize];
    std::memset(input_buffer_, 0, kWindowSize * sizeof(float));
    std::memset(output_buffer_, 0, kWindowSize * sizeof(float));

    dtln_freq_output_ = new float[kFftForTensorSize];
    dtln_time_output_ = new float[kWindowSize];
    std::memset(dtln_freq_output_, 0, kFftForTensorSize * sizeof(float));
    std::memset(dtln_time_output_, 0, kWindowSize * sizeof(float));

    input_mag_ = new float[kFftForTensorSize];
    input_phase_ = new float[kFftForTensorSize];
    estimated_block_ = new float[kWindowSize];
    std::memset(input_mag_, 0, kFftForTensorSize * sizeof(float));
    std::memset(input_phase_, 0, kFftForTensorSize * sizeof(float));
    std::memset(estimated_block_, 0, kWindowSize * sizeof(float));

    for (int i = 0; i < kNumModels; ++i) {
      states_[i] = new float[state_size_[i]];
      std::memset(states_[i], 0, state_size_[i] * sizeof(float));
    }

    input_sample_ = new float[kWindowSize];
    output_sample_ = new float[kWindowSize];
    std::memset(input_sample_, 0, kWindowSize * sizeof(float));
    std::memset(output_sample_, 0, kWindowSize * sizeof(float));

    init_success_ = true;
    ret = kWindowSize;
  } while (false);

  return ret;
}

void DTLN_NR::Impl::Release() {
  for (int i = 0; i < kNumModels; ++i) {
    if (tflite_model_[i] != nullptr) {
      TfLiteModelDelete(tflite_model_[i]);
      tflite_model_[i] = nullptr;
    }

    if (interpreter_[i] != nullptr) {
      TfLiteInterpreterDelete(interpreter_[i]);
      interpreter_[i] = nullptr;
    }
  }

  if (interpreter_options_ != nullptr) {
    TfLiteInterpreterOptionsDelete(interpreter_options_);
    interpreter_options_ = nullptr;
  }

  if (fftr_cfg_ != nullptr) {
    kiss_fft_free(fftr_cfg_);
    fftr_cfg_ = nullptr;
  }

  if (iffter_cfg_ != nullptr) {
    kiss_fft_free(iffter_cfg_);
    iffter_cfg_ = nullptr;
  }

  delete[] input_cpx_;
  input_cpx_ = nullptr;
  delete[] output_cpx_;
  output_cpx_ = nullptr;

  delete[] input_buffer_;
  input_buffer_ = nullptr;
  delete[] output_buffer_;
  output_buffer_ = nullptr;

  delete[] dtln_freq_output_;
  dtln_freq_output_ = nullptr;
  delete[] dtln_time_output_;
  dtln_time_output_ = nullptr;

  for (int i = 0; i < kNumModels; ++i) {
    delete[] states_[i];
    states_[i] = nullptr;
  }

  delete[] input_mag_;
  input_mag_ = nullptr;
  delete[] input_phase_;
  input_phase_ = nullptr;
  delete[] estimated_block_;
  estimated_block_ = nullptr;

  delete[] input_sample_;
  input_sample_ = nullptr;
  delete[] output_sample_;
  output_sample_ = nullptr;

  init_success_ = false;
}

int DTLN_NR::Impl::Process(short* input_buffer, short* output_buffer) {
  if (!init_success_ || input_buffer == nullptr || output_buffer == nullptr) {
    return -1;
  }

  for (int i = 0; i < kWindowSize; ++i) {
    input_sample_[i] = static_cast<float>(input_buffer[i]) / SHRT_MAX;
  }

  Denoise();

  for (int i = 0; i < kWindowSize; ++i) {
    output_buffer[i] = static_cast<short>(output_sample_[i] * SHRT_MAX);
  }

  return 0;
}

void DTLN_NR::Impl::Denoise() {
  const int num_blocks = kWindowSize / kWindowShift;

  float* input_sample = input_sample_;
  float* output_sample = output_sample_;

  for (int i = 0; i < num_blocks; ++i) {
    std::memmove(input_buffer_, input_buffer_ + kWindowShift,
                 (kWindowSize - kWindowShift) * sizeof(float));
    std::memcpy(input_buffer_ + (kWindowSize - kWindowShift), input_sample,
                kWindowShift * sizeof(float));

    std::memset(input_mag_, 0, kFftForTensorSize * sizeof(float));
    std::memset(input_phase_, 0, kFftForTensorSize * sizeof(float));
    std::memset(estimated_block_, 0, kWindowSize * sizeof(float));

    kiss_fftr(fftr_cfg_, input_buffer_, input_cpx_);

    for (int j = 0; j < kFftForTensorSize; ++j) {
      input_mag_[j] = std::sqrt(input_cpx_[j].r * input_cpx_[j].r +
                                input_cpx_[j].i * input_cpx_[j].i);
      input_phase_[j] = std::atan2(input_cpx_[j].i, input_cpx_[j].r);
    }

    TfLiteTensorCopyFromBuffer(input_tensor_[0][0], input_mag_,
                               kFftForTensorSize * sizeof(float));
    TfLiteTensorCopyFromBuffer(input_tensor_[0][1], states_[0],
                               state_size_[0] * sizeof(float));
    TfLiteInterpreterInvoke(interpreter_[0]);

    TfLiteTensorCopyToBuffer(output_tensor_[0][0], dtln_freq_output_,
                             kFftForTensorSize * sizeof(float));
    TfLiteTensorCopyToBuffer(output_tensor_[0][1], states_[0],
                             state_size_[0] * sizeof(float));

    for (int j = 0; j < kFftForTensorSize; ++j) {
      output_cpx_[j].r = input_mag_[j] * std::cos(input_phase_[j]) *
                         dtln_freq_output_[j];
      output_cpx_[j].i = input_mag_[j] * std::sin(input_phase_[j]) *
                         dtln_freq_output_[j];
    }

    kiss_fftri(iffter_cfg_, output_cpx_, estimated_block_);

    for (int j = 0; j < kWindowSize; ++j) {
      estimated_block_[j] /= kWindowSize;
    }

    TfLiteTensorCopyFromBuffer(input_tensor_[1][0], estimated_block_,
                               kWindowSize * sizeof(float));
    TfLiteTensorCopyFromBuffer(input_tensor_[1][1], states_[1],
                               state_size_[1] * sizeof(float));
    TfLiteInterpreterInvoke(interpreter_[1]);

    TfLiteTensorCopyToBuffer(output_tensor_[1][0], dtln_time_output_,
                             kWindowSize * sizeof(float));
    TfLiteTensorCopyToBuffer(output_tensor_[1][1], states_[1],
                             state_size_[1] * sizeof(float));

    std::memmove(output_buffer_, output_buffer_ + kWindowShift,
                 (kWindowSize - kWindowShift) * sizeof(float));
    std::memset(output_buffer_ + (kWindowSize - kWindowShift), 0,
                kWindowShift * sizeof(float));

    for (int j = 0; j < kWindowSize; ++j) {
      output_buffer_[j] += dtln_time_output_[j];
    }

    std::memcpy(output_sample, output_buffer_, kWindowShift * sizeof(float));

    input_sample += kWindowShift;
    output_sample += kWindowShift;
  }
}

DTLN_NR::DTLN_NR() : impl_(new Impl) {}

DTLN_NR::~DTLN_NR() {
  impl_->Release();
  delete impl_;
  impl_ = nullptr;
}

int DTLN_NR::Init() { return impl_->Init(); }

int DTLN_NR::Process(short* input_buffer, short* output_buffer) {
  return impl_->Process(input_buffer, output_buffer);
}
