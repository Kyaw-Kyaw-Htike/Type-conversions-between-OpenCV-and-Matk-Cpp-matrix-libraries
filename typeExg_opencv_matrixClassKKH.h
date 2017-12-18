#ifndef TYPEEXG_OPENCV_MATRIXCLASSKKH_H
#define TYPEEXG_OPENCV_MATRIXCLASSKKH_H

// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

#include "matrix_class_KKH.h"
#include "opencv2/opencv.hpp"
#include <cstring> // for memcpy
#include <cstdio> // printf

namespace hpers_TEOpencvMatKKH
{
	// convert typename and nchannels to opencv mat type such as CV_32FC1
	template <typename T>
	int getOpencvType(int nchannels)
	{
		int depth = cv::DataType<T>::depth;
		return (CV_MAT_DEPTH(depth) + (((nchannels)-1) << CV_CN_SHIFT));
	}
}

// work for opencv 2D matrices with any number of channels
// T can be any C++ native type 
template <typename T, int nchannels_>
void opencv2matKKH(const cv::Mat &matIn, Matk<T> &matOut)
{
	int nrows = matIn.rows;
	int ncols = matIn.cols;		
	int nchannels = matIn.channels();	
		
	if(matIn.dims != 2)
	{
		printf("Opencv Error: matIn.dims must be 2 (although it can have any number of channels.\n");
		return;
	}
	
	if(nchannels != nchannels_)
	{
		printf("Error: nchannels is different from template channels_ parameter.");
		return;
	}					

	matOut.create(nrows, ncols, nchannels);
	T * ptr_out = matOut.get_ptr();
	unsigned long count = 0;
	
	for (int k = 0; k < nchannels; k++)
		for (int j = 0; j < ncols; j++)
			for (int i = 0; i < nrows; i++)
				ptr_out[count++] = matIn.at<cv::Vec<T, nchannels_>>(i, j)[k];
}

// work for opencv 2D matrices with any number of channels
// T can be any C++ native type 
template <typename T, nchannels_>
void mat2KKH2opencv(const Matk<T> &matIn, cv::Mat &matOut)
{
	int nrows = matIn.nrows();
	int ncols = matIn.ncols();
	int nchannels = matIn.nchannels();
	
	if(nchannels != nchannels_)
	{
		printf("Error: nchannels is different from template channels_ parameter.");
		return;
	}

	matOut.create(nrows, ncols, hpers_TEOpencvArma::getOpencvType<T>(nchannels));
	
	T *ptr_in = matIn.get_ptr();
				
	for (int k = 0; k < nchannels; k++)
		for (int j = 0; j < ncols; j++)
			for (int i = 0; i < nrows; i++)
				matOut.at<cv::Vec<T, nchannels_>>(i, j)[k] = ptr_in[count++];	
			
}

// simply wrap an opencv 2D matrix (with any number of channels)
// in Matkr without any copying. Makr will NOT manage the memory;
// it just maintains a pointer to the data from opencv matrix.
// it's opencv's job to free it.
// ASSUMES that the opencv matrix data is contiguous and no gaps
template <typename T>
void opencv2matKKH_wrap(const cv::Mat &matIn, Matkr<T> matOut)
{
	matOut.wrap(matIn.rows, matIn.cols, matIn.channels(), matOut.ptr(0));
}

// simply wrap an Matkr matrix (with any number of channels)
// in opencv without any copying. Opencv will NOT manage the memory;
// it just maintains a pointer to the data from Matkr matrix.
// it's Matkr's job to free it.
template <typename T, int nchannels_>
cv::Mat matKKH2opencv_wrap(Matk<T> &matIn)
{
	return cv::Mat(matIn.nrows(), matIn.ncols(), CV_8UC(matIn.nchannels()), matIn.get_ptr());
}



#endif