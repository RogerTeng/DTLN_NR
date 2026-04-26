#include <cstdio>
#include <memory>
#include <string>

#include "DTLN_NR.h"

namespace {
constexpr int kWaveHeaderSize = 44;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    return 1;
  }

  const std::string input_rec_wave = argv[1];
  const std::string output_wave = argv[2];

  FILE* input_rec_file = std::fopen(input_rec_wave.c_str(), "rb");
  FILE* output_file = std::fopen(output_wave.c_str(), "wb+");
  if (input_rec_file == nullptr || output_file == nullptr) {
    if (input_rec_file != nullptr) {
      std::fclose(input_rec_file);
    }
    if (output_file != nullptr) {
      std::fclose(output_file);
    }
    return 1;
  }

  DTLN_NR dtln;
  const int frame_size = dtln.Init();
  if (frame_size <= 0) {
    std::fclose(input_rec_file);
    std::fclose(output_file);
    return 1;
  }

  std::unique_ptr<short[]> input_rec_sample(new short[frame_size]);
  std::unique_ptr<short[]> output_sample(new short[frame_size]);

  std::fread(input_rec_sample.get(), 1, kWaveHeaderSize, input_rec_file);

  while (true) {
    const int read_size =
        std::fread(input_rec_sample.get(), 1, frame_size * sizeof(short),
                   input_rec_file);
    if (read_size <= 0) {
      break;
    }

    dtln.Process(input_rec_sample.get(), output_sample.get());

    std::fwrite(output_sample.get(), 1, frame_size * sizeof(short), output_file);
  }

  std::fclose(input_rec_file);
  std::fclose(output_file);

  return 0;
}
