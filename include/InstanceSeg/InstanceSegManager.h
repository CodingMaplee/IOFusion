//
// Created by user on 2023/2/20.
//

#ifndef MAINLOOP_INSTANCESEGMANAGER_H
#define MAINLOOP_INSTANCESEGMANAGER_H

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include <atomic>
#include <thread>
#include <queue>
#include "CUDAImageUtil.h"
#include "MatrixConversion.h"

class InstanceSegManager {
public:
    InstanceSegManager(uint width, uint height);
    ~InstanceSegManager();
    void initialise();
    void* loadModule();

    // Warning, getPyObject requiers a decref:
    inline PyObject* getPyObject(const char* name);

    void extractInstanceResult(int** pData);

    PyObject* createArguments(cv::Mat rgbImage);

    void executeSegmentation(cv::Mat imgTar, int** pData);

    void setMaskMap(uchar* maskGPU, uint width, uint height);
    uchar* getInstanceMapGpu();
    uchar* getInstanceMapCpu();
    void setPersonMap(uchar* dynamicMap);
    void setObjMap(uchar* dynamicMap);
    uchar* getPersonMapCpu();
    uchar* getPersonMapGpu();
    uchar* getObjMapCpu();
    void updateExistingDynamicPx_currFrame(const float* flowUGpu, const float* flowVGpu);
    void genPersonMask();
private:
    PyObject *pModule = NULL;
    PyObject *pExecute = NULL;

    // For parallel execution
    std::thread thread;
    std::exception_ptr threadException;

    uchar* g_currMaskMapGpu;
    uchar* g_currPersonMaskGpu;
    uchar* g_currObjMaskGpu;
    uint g_maskMapWidth;
    uint g_maskMapHeight;

    std::vector<float2*> g_dynamicPX;
    std::vector<uint> g_dynamicCatLabel;

};




#endif //MAINLOOP_INSTANCESEGMANAGER_H
