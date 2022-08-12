# DTLN_NR
A stand alone wrapper library in C code of [DTLN](https://github.com/breizhn/DTLN).<br/>
Inlcude prebuilt TensorFlow Lite v2.5.2 for Windows x64/macOS(Intel)/macOS(Apple Silicon).

If you need AEC, please check [DTLN_AEC](https://github.com/RogerTeng/DTLN_AEC).

#### This repository provide VS2019 project, but you could build this source code in xCode.

I use hardcode model in this project, if you want change model,<br/>
Use
	
	TfLiteModelCreateFromFile()


To replace
	
	TfLiteModelCreate()

## Acknowledgement
* This project is based on the [DTLN](https://github.com/breizhn/DTLN) by [breizhn](https://github.com/breizhn).
* FFT from [KISS FFT](https://github.com/mborgerding/kissfft) by [mborgerding](https://github.com/mborgerding).

## Citing

```BibTex
@inproceedings{Westhausen2020,
  author={Nils L. Westhausen and Bernd T. Meyer},
  title={{Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={2477--2481},
  doi={10.21437/Interspeech.2020-2631},
  url={http://dx.doi.org/10.21437/Interspeech.2020-2631}
}
