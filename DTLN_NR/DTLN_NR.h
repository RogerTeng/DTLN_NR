#ifndef DTLN_NR_DTLN_NR_H_
#define DTLN_NR_DTLN_NR_H_

#if defined(_WIN32) || defined(_WIN64)
#ifdef DLTNNR_EXPORTS
#define DLTNNR __declspec(dllexport)
#else
#define DLTNNR __declspec(dllimport)
#endif

class DLTNNR DTLN_NR
#else
class DTLN_NR
#endif
{
 public:
  DTLN_NR();
  ~DTLN_NR();

  // Returns the number of input samples per frame; -1 on failure.
  int Init();

  // Returns 0 on success; -1 on failure.
  int Process(short* input_buffer, short* output_buffer);

 private:
  class Impl;
  Impl* impl_ = nullptr;
};

#endif  // DTLN_NR_DTLN_NR_H_
