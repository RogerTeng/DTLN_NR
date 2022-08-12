
#if defined(_WIN32) || defined(_WIN64)

#ifdef DLTNNR_EXPORTS
#    define DLTNNR __declspec(dllexport)
#else
#    define DLTNNR __declspec(dllimport)
#endif

//Only support 16K 16Bit Mono PCM

class DLTNNR DTLN_NR
//Windows win32/x86_64
#else
class DTLN_NR //#elif defined(__APPLE__)
//macOS
#endif
{
public:
	DTLN_NR();
	~DTLN_NR();

	//Return number of input samples, -1 = Fail
	int Init(void);

	//0 = Success, -1 = Fail
	int Process(short *lpsInputBuffer, short *lpsOutputBuffer);

private:
	class m_Impl;
	m_Impl *m_lpoImpl = nullptr;
};


