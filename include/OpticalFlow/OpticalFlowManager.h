//
// Created by maple on 22-6-1.
//
#pragma once
#ifndef MAINLOOP_RAFT_H
#define MAINLOOP_RAFT_H

#endif //MAINLOOP_RAFT_H

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include <atomic>
#include <thread>
#include <queue>
#include "CUDAImageUtil.h"
#include "MatrixConversion.h"


class OpticalFlowManager {
public:
    OpticalFlowManager(uint width, uint height);
    ~OpticalFlowManager();
    void initialise();
    void* loadModule();

    // Warning, getPyObject requiers a decref:
    inline PyObject* getPyObject(const char* name);

    void extractOpticalFlowResult(float** pData_u, float** pData_v);

    PyObject* createArguments(cv::Mat rgbImage);

    void executeOpticalFlow(cv::Mat imgSrc, cv::Mat imgTar, float** pData_u, float** pData_v);

    uchar* extractMotionConsistency(float* consistency_float_px_cpu, float threshold, const uchar* existingDynamicPx, const float* depTar, const float* depSrc, const uchar4* colorTar, const uchar4* colorSrc,
                                   const float* flowU, const float* flowV, const uint width, const uint height,
                                   Eigen::Matrix4f depth_intrinsics, Eigen::Matrix4f transformLast, Eigen::Matrix4f transformCurrent, uint currFrameNumber);

    void setOpticalFlowMap(float* flowU, float* flowV, uint width, uint height);
    void getOpticalFlowMapGpu(float* flowUGpu, float* flowVGpu);

private:
    PyObject *pModule = NULL;
    PyObject *pExecute = NULL;

    // For parallel execution
    std::thread thread;
    std::exception_ptr threadException;

    float* g_opticalFlowUGpu;
    float* g_opticalFlowVGpu;
    float* g_opticalFlowU;
    float* g_opticalFlowV;

    uint g_opticalFlowWidth;
    uint g_opticalFlowHeight;
};