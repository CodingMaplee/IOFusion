//
// Created by maple on 22-6-5.
//

#include <OpticalFlow/OpticalFlowManager.h>
#include <fstream>
#include <string>
#include <opencv2/imgcodecs.hpp>




OpticalFlowManager::OpticalFlowManager(uint width, uint height)
{
    initialise();

    MLIB_CUDA_SAFE_CALL(cudaMalloc(&g_opticalFlowUGpu, sizeof(float) * width * height));
    MLIB_CUDA_SAFE_CALL(cudaMalloc(&g_opticalFlowVGpu, sizeof(float) * width * height));
}
OpticalFlowManager::~OpticalFlowManager(){
    Py_XDECREF(pModule);
    Py_XDECREF(pExecute);
    Py_Finalize();
}

void OpticalFlowManager::initialise(){
    std::cout << "* Initialising Optical Flow (thread: " << std::this_thread::get_id() << ") ..." << std::endl;
    Py_SetProgramName((wchar_t*)L"Raft");
    Py_Initialize();
    wchar_t const * argv2[] = { L"Raft.py" };
    PySys_SetArgv(1, const_cast<wchar_t**>(argv2));
    // Load module
    loadModule();
    // Get function
    pExecute = PyObject_GetAttrString(pModule, "execute");
    if(pExecute == NULL || !PyCallable_Check(pExecute)) {
        if(PyErr_Occurred()) {
            std::cout << "Python error indicator is set:" << std::endl;
            PyErr_Print();
        }
        throw std::runtime_error("Could not load function 'execute' from python module.");
    }
    std::cout << "* Initialised Optical Flow" << std::endl;

    //Py_XDECREF(PyObject_CallFunctionObjArgs(pExecute, NULL));
}

void* OpticalFlowManager::loadModule(){
    std::cout << " * Loading module..." << std::endl;
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../include/OpticalFlow')");
    pModule = PyImport_ImportModule("Gmflow");
    if(pModule == NULL) {
        if(PyErr_Occurred()) {
            std::cout << "Python error indicator is set:" << std::endl;
            PyErr_Print();
        }
        throw std::runtime_error("Could not open optical flow module.");
    }
    import_array();
    return 0;
}

inline PyObject* OpticalFlowManager::getPyObject(const char* name){
    PyObject* obj = PyObject_GetAttrString(pModule, name);
    if(!obj || obj == Py_None) throw std::runtime_error(std::string("Failed to get python object: ") + name);
    return obj;
}

void OpticalFlowManager::extractOpticalFlowResult(float** pData_u, float** pData_v){

    PyObject* p_flow_up_u = getPyObject("flow_up_u");
    PyObject* p_flow_up_v = getPyObject("flow_up_v");

    PyArrayObject *pFlow2Array_u = (PyArrayObject*)(p_flow_up_u);
    PyArrayObject *pFlow2Array_v = (PyArrayObject*)(p_flow_up_v);

    *pData_u = (float*)PyArray_GETPTR1(pFlow2Array_u,0);
    *pData_v = (float*)PyArray_GETPTR1(pFlow2Array_v,0);

    //npy_intp h = PyArray_DIM(pFlow2Array,0);
    //npy_intp w = PyArray_DIM(pFlow2Array,1);

    //cv::Mat result;
    //cv::Mat(h,w, CV_8UC1, pData).copyTo(result);
    Py_DECREF(p_flow_up_u);
    Py_DECREF(p_flow_up_v);

    //return result;
}

PyObject* OpticalFlowManager::createArguments(cv::Mat rgbImage){
    assert(rgbImage.channels() == 3);
    npy_intp dims[3] = { rgbImage.rows, rgbImage.cols, 3 };
    return PyArray_SimpleNewFromData(3, dims, NPY_UINT8, rgbImage.data); // TODO Release?
}

void OpticalFlowManager::executeOpticalFlow(cv::Mat imgSrc, cv::Mat imgTar, float** pData_u, float** pData_v) //
{
    Py_XDECREF(PyObject_CallFunctionObjArgs(pExecute, createArguments(imgSrc), createArguments(imgTar), NULL));
    //std::cout<<"kkk"<<std::endl;
    extractOpticalFlowResult(pData_u, pData_v);
}
void OpticalFlowManager::setOpticalFlowMap(float* flowU, float* flowV, uint width, uint height) //cv::Mat
{

    MLIB_CUDA_SAFE_CALL(cudaMemcpy(g_opticalFlowUGpu, flowU, sizeof(float) * width * height, cudaMemcpyHostToDevice));
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(g_opticalFlowVGpu, flowV, sizeof(float) * width * height, cudaMemcpyHostToDevice));
    g_opticalFlowWidth = width;
    g_opticalFlowHeight = height;
}

void OpticalFlowManager::getOpticalFlowMapGpu(float* flowUGpu, float* flowVGpu)
{
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(flowUGpu, g_opticalFlowUGpu, sizeof(float) * g_opticalFlowWidth * g_opticalFlowHeight, cudaMemcpyDeviceToDevice));
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(flowVGpu, g_opticalFlowVGpu, sizeof(float) * g_opticalFlowWidth * g_opticalFlowHeight, cudaMemcpyDeviceToDevice));
}
uchar* OpticalFlowManager::extractMotionConsistency(float* consistency_float_px_cpu, float threshold, const uchar* existingDynamicPx, const float* depTar, const float* depSrc, const uchar4* colorTar, const uchar4* colorSrc,
                                            const float* flowU, const float* flowV, const uint width, const uint height,
                                            Eigen::Matrix4f depth_intrinsics, Eigen::Matrix4f transformLast, Eigen::Matrix4f transformCurrent,uint currFrameNumber)
{
    uchar* consistency_optical_flow_px;
    MLIB_CUDA_SAFE_CALL(cudaMalloc(&consistency_optical_flow_px, sizeof(uchar) * width * height));
    MLIB_CUDA_SAFE_CALL(cudaMemset(consistency_optical_flow_px, 0, sizeof(uchar) * width * height));
    float* consistency_float_px;
    MLIB_CUDA_SAFE_CALL(cudaMalloc(&consistency_float_px, sizeof(float) * width * height));
    float* intensityTar;
    MLIB_CUDA_SAFE_CALL(cudaMalloc(&intensityTar, sizeof(float) * width * height));
    CUDAImageUtil::convertUCHAR4ToIntensityFloat(intensityTar, colorTar, width, height);

    float* intensitySrc;
    MLIB_CUDA_SAFE_CALL(cudaMalloc(&intensitySrc, sizeof(float) * width * height));
    CUDAImageUtil::convertUCHAR4ToIntensityFloat(intensitySrc, colorSrc, width, height);

    CUDAImageUtil::extractMotionConsistency(consistency_float_px, consistency_optical_flow_px,
                                            threshold, existingDynamicPx, depTar, depSrc,
                                            intensityTar, intensitySrc,
                                            flowU, flowV, depth_intrinsics,
                                            transformLast, transformCurrent.inverse(), width, height);
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(consistency_float_px_cpu, consistency_float_px, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
    cv::Mat flowResidual = cv::Mat(height, width, CV_8UC3);
    for(int i = 0;i<height;i++)
    {
        for(int j = 0;j<width;j++)
        {
            float consis = consistency_float_px_cpu[j + i * width];
            flowResidual.at<cv::Vec3b>(i, j)[0] = 255 * consis / 15;
            flowResidual.at<cv::Vec3b>(i, j)[1] = 255 * consis / 15;
            flowResidual.at<cv::Vec3b>(i, j)[2] = 255 * consis / 15;
        }
    }
    cv::imwrite("../FlowResidual/"+std::to_string(currFrameNumber)+".png", flowResidual);
    MLIB_CUDA_SAFE_CALL(cudaFree(intensityTar));
    MLIB_CUDA_SAFE_CALL(cudaFree(intensitySrc));
    MLIB_CUDA_SAFE_CALL(cudaFree(consistency_float_px));
    return consistency_optical_flow_px;
}

