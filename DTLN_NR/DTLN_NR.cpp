
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/c/c_api.h>

#include "model_1.h"
#include "model_2.h"

//Use KissFFT https://github.com/mborgerding/kissfft
#include "kiss_fftr.h"

#include "DTLN_NR.h"

//1 Network contain 2 models
//Please check : https://github.com/breizhn/DTLN
//This code is translate from : https://github.com/breizhn/DTLN/blob/master/real_time_processing_tf_lite.py

//const param
constexpr auto k_nWindowSize = 512;
constexpr auto k_nWindowShift = 128;
constexpr auto k_nFftForTensorSize = (k_nWindowSize / 2 + 1);

constexpr auto k_nNumModels = 2;

constexpr auto k_nNumThreads = 1;

class DTLN_NR::m_Impl
{
public:
	
	int Init(void);
	void Release(void);

	int Process(short *lpsInputBuffer, short *lpsOutputBuffer);
	void Denoise(void);

	TfLiteModel *m_lppoTfliteModel[k_nNumModels];
	TfLiteInterpreter *m_lppoInterpreter[k_nNumModels];
	TfLiteInterpreterOptions *m_lpoInterpreterOptions = nullptr;

	TfLiteTensor *m_lppoInputTensor[k_nNumModels][2];
	const TfLiteTensor *m_lppoOutputTensor[k_nNumModels][2];

	//FFT
	kiss_fftr_cfg m_lpoFftrCfg = nullptr;
	kiss_fftr_cfg m_lpoIfftrCfg = nullptr;

	kiss_fft_cpx *m_lpoInputCpx = nullptr;
	kiss_fft_cpx *m_lpoOutputCpx = nullptr;

	//Internal buffer
	float *m_lpfInputBuffer = nullptr;
	float *m_lpfOutputBuffer = nullptr;

	float *m_lpfDtlnFreqOutput = nullptr;
	float *m_lpfDtlnTimeOutput = nullptr;

	int m_lpnStateSize[k_nNumModels];
	float *m_lppfStates[k_nNumModels];

	float *m_lpfInputMag = nullptr;
	float *m_lpfInputPhase = nullptr;
	float *m_lpfEstimatedBlock = nullptr;

	//Format change buffer
	float *m_lpfInputSample = nullptr;
	float *m_lpfOutputSample = nullptr;

	bool m_bInitSuccess = false;
		
};

int DTLN_NR::m_Impl::Init(void)
{
	int nRet = -1;

	do 
	{
		for (int i = 0; i < k_nNumModels; i++)
		{
			this->m_lppoTfliteModel[i] = nullptr;
			this->m_lppoInterpreter[i] = nullptr;

			this->m_lppfStates[i] = nullptr;
		}

		//Load models
		this->m_lppoTfliteModel[0] = TfLiteModelCreate(k_lpszModel1Tflite, k_nModel1TfliteLen);
		this->m_lppoTfliteModel[1] = TfLiteModelCreate(k_lpszModel2Tflite, k_nModel2TfliteLen);

		if (this->m_lppoTfliteModel[0] == nullptr || this->m_lppoTfliteModel[1] == nullptr)
			break;

		//Create option
		this->m_lpoInterpreterOptions = TfLiteInterpreterOptionsCreate();
		TfLiteInterpreterOptionsSetNumThreads(this->m_lpoInterpreterOptions, k_nNumThreads);

		//Create the interpreter
		this->m_lppoInterpreter[0] = TfLiteInterpreterCreate(this->m_lppoTfliteModel[0], this->m_lpoInterpreterOptions);
		this->m_lppoInterpreter[1] = TfLiteInterpreterCreate(this->m_lppoTfliteModel[1], this->m_lpoInterpreterOptions);

		if (this->m_lppoInterpreter[0] == nullptr || this->m_lppoInterpreter[1] == nullptr)
			break;

		//Allocate tensor
		if (TfLiteInterpreterAllocateTensors(this->m_lppoInterpreter[0]) != kTfLiteOk)
			break;
		if (TfLiteInterpreterAllocateTensors(this->m_lppoInterpreter[1]) != kTfLiteOk)
			break;

		for (int i = 0; i < k_nNumModels; i++)
		{
			this->m_lppoInputTensor[i][0] = TfLiteInterpreterGetInputTensor(this->m_lppoInterpreter[i], 0);
			this->m_lppoInputTensor[i][1] = TfLiteInterpreterGetInputTensor(this->m_lppoInterpreter[i], 1);

			this->m_lppoOutputTensor[i][0] = TfLiteInterpreterGetOutputTensor(this->m_lppoInterpreter[i], 0);
			this->m_lppoOutputTensor[i][1] = TfLiteInterpreterGetOutputTensor(this->m_lppoInterpreter[i], 1);

			this->m_lpnStateSize[i] = this->m_lppoInputTensor[i][1]->bytes / sizeof(float);
		}

		//RFFT/iRFFT
		this->m_lpoFftrCfg = kiss_fftr_alloc(k_nWindowSize, 0, 0, 0);
		this->m_lpoIfftrCfg = kiss_fftr_alloc(k_nWindowSize, 1, 0, 0);
	
		this->m_lpoInputCpx = new kiss_fft_cpx[k_nFftForTensorSize];
		this->m_lpoOutputCpx = new kiss_fft_cpx[k_nFftForTensorSize];

		//Internal buffer
		this->m_lpfInputBuffer = new float[k_nWindowSize];
		this->m_lpfOutputBuffer = new float[k_nWindowSize];

		memset(this->m_lpfInputBuffer, 0, k_nWindowSize * sizeof(float));
		memset(this->m_lpfOutputBuffer, 0, k_nWindowSize * sizeof(float));

		this->m_lpfDtlnFreqOutput = new float[k_nFftForTensorSize];
		this->m_lpfDtlnTimeOutput = new float[k_nWindowSize];

		memset(this->m_lpfDtlnFreqOutput, 0, k_nFftForTensorSize * sizeof(float));
		memset(this->m_lpfDtlnTimeOutput, 0, k_nWindowSize * sizeof(float));

		this->m_lpfInputMag = new float[k_nFftForTensorSize];
		this->m_lpfInputPhase = new float[k_nFftForTensorSize];
		this->m_lpfEstimatedBlock = new float[k_nWindowSize];

		memset(this->m_lpfInputMag, 0, k_nFftForTensorSize * sizeof(float));
		memset(this->m_lpfInputPhase, 0, k_nFftForTensorSize * sizeof(float));
		memset(this->m_lpfEstimatedBlock, 0, k_nWindowSize * sizeof(float));

		for (int i = 0; i < k_nNumModels; i++)
		{
			this->m_lppfStates[i] = new float[this->m_lpnStateSize[i]];
			memset(this->m_lppfStates[i], 0, this->m_lpnStateSize[i] * sizeof(float));
		}
		
		//Format change buffer
		this->m_lpfInputSample = new float[k_nWindowSize];
		this->m_lpfOutputSample = new float[k_nWindowSize];

		memset(this->m_lpfInputSample, 0, k_nWindowSize * sizeof(float));
		memset(this->m_lpfOutputSample, 0, k_nWindowSize * sizeof(float));

		this->m_bInitSuccess = true;

		nRet = k_nWindowSize;

	}
	while (0);

	return nRet;
}

void DTLN_NR::m_Impl::Release(void)
{
	//Tensorflow lite
	for (int i = 0; i < k_nNumModels; i++)
	{
		if (this->m_lppoTfliteModel[i] != nullptr)
			TfLiteModelDelete(this->m_lppoTfliteModel[i]);

		if (this->m_lppoInterpreter[i] != nullptr)
			TfLiteInterpreterDelete(this->m_lppoInterpreter[i]);
	}

	if (this->m_lpoInterpreterOptions != nullptr)
		TfLiteInterpreterOptionsDelete(this->m_lpoInterpreterOptions);

	//RFFT/iRFFT
	if (this->m_lpoFftrCfg != nullptr)
		kiss_fft_free(this->m_lpoFftrCfg);

	if (this->m_lpoIfftrCfg != nullptr)
		kiss_fft_free(this->m_lpoIfftrCfg);

	if (this->m_lpoInputCpx != nullptr)
		delete[] this->m_lpoInputCpx;

	if (this->m_lpoOutputCpx != nullptr)
		delete[] this->m_lpoOutputCpx;


	//Internal buffer
	if (this->m_lpfInputBuffer != nullptr)
		delete[] this->m_lpfInputBuffer;

	if (this->m_lpfOutputBuffer != nullptr)
		delete[] this->m_lpfOutputBuffer;

	if (this->m_lpfDtlnFreqOutput != nullptr)
		delete[] this->m_lpfDtlnFreqOutput;

	if (this->m_lpfDtlnTimeOutput != nullptr)
		delete[] this->m_lpfDtlnTimeOutput;

	for (int i = 0; i < k_nNumModels; i++)
	{
		if (this->m_lppfStates[i] != nullptr)
			delete[] this->m_lppfStates[i];
	}

	if (this->m_lpfInputMag != nullptr)
		delete[] this->m_lpfInputMag;

	if (this->m_lpfInputPhase != nullptr)
		delete[] this->m_lpfInputPhase;

	if (this->m_lpfEstimatedBlock != nullptr)
		delete[] this->m_lpfEstimatedBlock;

	//Format change buffer
	if (this->m_lpfInputSample != nullptr)
		delete[] this->m_lpfInputSample;

	if (this->m_lpfOutputSample != nullptr)
		delete[] this->m_lpfOutputSample;
}

int DTLN_NR::m_Impl::Process(short *lpsInputBuffer, short *lpsOutputBuffer)
{
	int nRet = -1;

	do
	{
		if (this->m_bInitSuccess == false)
			break;

		if (lpsInputBuffer == nullptr || lpsOutputBuffer == nullptr)
			break;

		//Convert short to float
		for (int i = 0; i < k_nWindowSize; i++)
		{
			this->m_lpfInputSample[i] = (float)lpsInputBuffer[i] * 1.0f / SHRT_MAX;
		}

		this->Denoise();

		//Convert float to short
		for (int i = 0; i < k_nWindowSize; i++)
		{
			lpsOutputBuffer[i] = (short)(this->m_lpfOutputSample[i] * SHRT_MAX);
		}

		nRet = 0;
	} 
	while (0);

	return nRet;
}

void DTLN_NR::m_Impl::Denoise(void)
{
	int nNumBlocks = k_nWindowSize / k_nWindowShift;

	float *pfInputSample = this->m_lpfInputSample;
	float *pfOutputSample = this->m_lpfOutputSample;

	for (int i = 0; i < nNumBlocks; i++)
	{
		//Buffer shift to match FFT size
		memmove(this->m_lpfInputBuffer, this->m_lpfInputBuffer + k_nWindowShift, (k_nWindowSize - k_nWindowShift) * sizeof(float));
		memcpy(this->m_lpfInputBuffer + (k_nWindowSize - k_nWindowShift), pfInputSample, k_nWindowShift * sizeof(float));
		
		//Prepare buffer
		memset(this->m_lpfInputMag, 0, k_nFftForTensorSize * sizeof(float));
		memset(this->m_lpfInputPhase, 0, k_nFftForTensorSize * sizeof(float));
		memset(this->m_lpfEstimatedBlock, 0, k_nWindowSize * sizeof(float));

		//Use RFFT/iRFFT to implement STFT/iSTFT

		//RFFT
		kiss_fftr(this->m_lpoFftrCfg, this->m_lpfInputBuffer, this->m_lpoInputCpx);

		//Calculate Mag/Phase
		for (int j = 0; j < k_nFftForTensorSize; j++)
		{
			//How to calculate Mag/Phase:
			//check 3a/3b in https://www.gaussianwaves.com/2015/11/interpreting-fft-results-obtaining-magnitude-and-phase-information/
			this->m_lpfInputMag[j] = sqrtf(this->m_lpoInputCpx[j].r * this->m_lpoInputCpx[j].r + this->m_lpoInputCpx[j].i * this->m_lpoInputCpx[j].i);
			this->m_lpfInputPhase[j] = atan2f(this->m_lpoInputCpx[j].i, this->m_lpoInputCpx[j].r);
		}

		//Set data into tensor
		TfLiteTensorCopyFromBuffer(this->m_lppoInputTensor[0][0], this->m_lpfInputMag, k_nFftForTensorSize * sizeof(float));
		TfLiteTensorCopyFromBuffer(this->m_lppoInputTensor[0][1], this->m_lppfStates[0], this->m_lpnStateSize[0] * sizeof(float));

		//DTLN for freq domain
		TfLiteInterpreterInvoke(this->m_lppoInterpreter[0]);

		//Get data from tensor
		TfLiteTensorCopyToBuffer(this->m_lppoOutputTensor[0][0], this->m_lpfDtlnFreqOutput, k_nFftForTensorSize * sizeof(float));
		TfLiteTensorCopyToBuffer(this->m_lppoOutputTensor[0][1], this->m_lppfStates[0], this->m_lpnStateSize[0] * sizeof(float));

		//iRFFT
		//this->m_lpfDtlnFreqOutput is out_mask
		//Use orignal Mag/Phase to restore generated freq
		for (int j = 0; j < k_nFftForTensorSize; j++)
		{
			//Re{ z } = Re{ a + ib } = Mag * cos[φ] * freq
			//Im{ z } = Im{ a + ib } = Mag * sin[φ] * freq
			this->m_lpoOutputCpx[j].r = this->m_lpfInputMag[j] * cosf(this->m_lpfInputPhase[j]) * this->m_lpfDtlnFreqOutput[j];
			this->m_lpoOutputCpx[j].i = this->m_lpfInputMag[j] * sinf(this->m_lpfInputPhase[j]) * this->m_lpfDtlnFreqOutput[j];
		}

		kiss_fftri(this->m_lpoIfftrCfg, this->m_lpoOutputCpx, this->m_lpfEstimatedBlock);

		//FFT coefficient 1/N
		for (int j = 0; j < k_nWindowSize; j++)
			this->m_lpfEstimatedBlock[j] = this->m_lpfEstimatedBlock[j] / k_nWindowSize;

		//Set data into tensor
		TfLiteTensorCopyFromBuffer(this->m_lppoInputTensor[1][0], this->m_lpfEstimatedBlock, k_nWindowSize * sizeof(float));
		TfLiteTensorCopyFromBuffer(this->m_lppoInputTensor[1][1], this->m_lppfStates[1], this->m_lpnStateSize[1] * sizeof(float));

		//DTLN for time domain
		TfLiteInterpreterInvoke(this->m_lppoInterpreter[1]);

		//Get data from tensor
		TfLiteTensorCopyToBuffer(this->m_lppoOutputTensor[1][0], this->m_lpfDtlnTimeOutput, k_nWindowSize * sizeof(float));
		TfLiteTensorCopyToBuffer(this->m_lppoOutputTensor[1][1], this->m_lppfStates[1], this->m_lpnStateSize[1] * sizeof(float));

		//Overlap add
		memmove(this->m_lpfOutputBuffer, this->m_lpfOutputBuffer + k_nWindowShift, (k_nWindowSize - k_nWindowShift) * sizeof(float));
		memset(this->m_lpfOutputBuffer + (k_nWindowSize - k_nWindowShift), 0, k_nWindowShift * sizeof(float));

		for (int j = 0; j < k_nWindowSize; j++)
			this->m_lpfOutputBuffer[j] += this->m_lpfDtlnTimeOutput[j];


		memcpy(pfOutputSample, this->m_lpfOutputBuffer, k_nWindowShift * sizeof(float));

		pfInputSample += k_nWindowShift;
		pfOutputSample += k_nWindowShift;
	}
}

DTLN_NR::DTLN_NR() :m_lpoImpl(new DTLN_NR::m_Impl)
{
}

DTLN_NR::~DTLN_NR()
{
	this->m_lpoImpl->Release();

	delete this->m_lpoImpl;
	this->m_lpoImpl = nullptr;
}

int DTLN_NR::Init(void)
{
	return this->m_lpoImpl->Init();
}

int DTLN_NR::Process(short *lpsInputBuffer, short *lpsOutputBuffer)
{
	return this->m_lpoImpl->Process(lpsInputBuffer, lpsOutputBuffer);
}

