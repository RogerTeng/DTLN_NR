

#include <string>
#include "DTLN_NR.h"

int main(int argc, char *argv[])
{
    //lpszInputRecWave is a recording data

    std::string lpszInputRecWave = std::string(argv[1]);
    std::string lpszOutputWave = std::string(argv[2]);

    FILE *lpoInputRecFile = NULL;
    FILE *lpoOutputFile = NULL;

    short *lpsInputRecSample = NULL;
    short *lpsOutputSample = NULL;

    int nReadSize, nFrameSize;

    lpoInputRecFile = fopen(lpszInputRecWave.c_str(), "rb");
    lpoOutputFile = fopen(lpszOutputWave.c_str(), "wb+");

    DTLN_NR oDtln;

    nFrameSize = oDtlnAec.Init();

    lpsInputRecSample = new short[nFrameSize];
    lpsOutputSample = new short[nFrameSize];

    //Skip wave header
    fread(lpsInputRecSample, 1, 44, lpoInputRecFile);

    while (true)
    {
        nReadSize = fread(lpsInputRecSample, 1, nFrameSize * sizeof(short), lpoInputRecFile);
        if (nReadSize <= 0)
            break;

        oDtln.Process(lpsInputRecSample, lpsOutputSample);

        //write PCM
        fwrite(lpsOutputSample, 1, nFrameSize * sizeof(short), lpoOutputFile);
    }

    fclose(lpoInputRecFile);
    fclose(lpoOutputFile);

    if (lpsInputRecSample != NULL)
        delete[] lpsInputRecSample;

    if (lpsOutputSample != NULL)
        delete[] lpsOutputSample;

    return 0;

}