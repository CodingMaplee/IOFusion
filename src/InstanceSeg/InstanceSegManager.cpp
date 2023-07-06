//
// Created by user on 2023/2/20.
//
#include <InstanceSeg/InstanceSegManager.h>
#include <fstream>
#include <string>
#include <opencv2/imgcodecs.hpp>


InstanceSegManager::InstanceSegManager(uint width, uint height)
{
    initialise();

    MLIB_CUDA_SAFE_CALL(cudaMalloc(&g_currMaskMapGpu, sizeof(int) * width * height));
    MLIB_CUDA_SAFE_CALL(cudaMalloc(&g_currPersonMaskGpu, sizeof(int) * width * height));
    MLIB_CUDA_SAFE_CALL(cudaMalloc(&g_currObjMaskGpu, sizeof(int) * width * height));
}
InstanceSegManager::~InstanceSegManager(){
    Py_XDECREF(pModule);
    Py_XDECREF(pExecute);
    Py_Finalize();
}

void InstanceSegManager::initialise(){
    std::cout << "* Initialising SOLO (thread: " << std::this_thread::get_id() << ") ..." << std::endl;

    Py_SetProgramName((wchar_t*)L"Solo");
    Py_Initialize();
    wchar_t const * argv2[] = { L"Solo.py" };
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
    std::cout << "* Initialised SOLO" << std::endl;

    //Py_XDECREF(PyObject_CallFunctionObjArgs(pExecute, NULL));
}

void* InstanceSegManager::loadModule(){
    std::cout << " * Loading module..." << std::endl;
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../include/InstanceSeg')");
    pModule = PyImport_ImportModule("Solo");
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

inline PyObject* InstanceSegManager::getPyObject(const char* name){
    PyObject* obj = PyObject_GetAttrString(pModule, name);
    if(!obj || obj == Py_None) throw std::runtime_error(std::string("Failed to get python object: ") + name);
    return obj;
}

void InstanceSegManager::extractInstanceResult(int** pData){
    PyObject* result = getPyObject("output");
    PyArrayObject *pResult = (PyArrayObject*)(result);
    *pData = (int*)PyArray_GETPTR1(pResult,0);
    //npy_intp h = PyArray_DIM(pFlow2Array,0);
    //npy_intp w = PyArray_DIM(pFlow2Array,1);

    //cv::Mat result;
    //cv::Mat(h,w, CV_8UC1, pData).copyTo(result);
    Py_DECREF(result);
    //return result;
}

PyObject* InstanceSegManager::createArguments(cv::Mat rgbImage){
    assert(rgbImage.channels() == 3);
    npy_intp dims[3] = { rgbImage.rows, rgbImage.cols, 3 };
    return PyArray_SimpleNewFromData(3, dims, NPY_UINT8, rgbImage.data); // TODO Release?
}

void InstanceSegManager::executeSegmentation(cv::Mat imgTar, int** pData) //
{
    Py_XDECREF(PyObject_CallFunctionObjArgs(pExecute, createArguments(imgTar), NULL));
    extractInstanceResult(pData);
}
void InstanceSegManager::setMaskMap(uchar* mask, uint width, uint height) //cv::Mat
{

    MLIB_CUDA_SAFE_CALL(cudaMemcpy(g_currMaskMapGpu, mask, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));
    g_maskMapWidth = width;
    g_maskMapHeight = height;
}
void InstanceSegManager::genPersonMask()
{
    if(g_currMaskMapGpu)
    {
        uchar* personMask;
        MLIB_CUDA_SAFE_CALL(cudaMalloc(&personMask, sizeof(uchar) * g_maskMapWidth * g_maskMapHeight));
        MLIB_CUDA_SAFE_CALL(cudaMemset(personMask, 0, sizeof(uchar) * g_maskMapWidth * g_maskMapHeight));
        CUDAImageUtil::genPersonMask(g_currMaskMapGpu, personMask, g_maskMapWidth, g_maskMapHeight);

        MLIB_CUDA_SAFE_CALL(cudaMemcpy(g_currPersonMaskGpu, personMask, sizeof(uchar) * g_maskMapWidth * g_maskMapHeight, cudaMemcpyDeviceToDevice));
        MLIB_CUDA_SAFE_CALL(cudaFree(personMask));
    }
}
uchar* InstanceSegManager::getInstanceMapGpu()
{
    //uchar* mask;
    //MLIB_CUDA_SAFE_CALL(cudaMalloc(&mask, sizeof(uchar) * g_maskMapWidth * g_maskMapHeight));
    //MLIB_CUDA_SAFE_CALL(cudaMemcpy(mask, g_currMaskMapGpu, sizeof(uchar) * g_maskMapWidth * g_maskMapHeight, cudaMemcpyDeviceToDevice));
    return g_currMaskMapGpu;
}
uchar* InstanceSegManager::getInstanceMapCpu()
{
    uchar* mask = new uchar[g_maskMapWidth * g_maskMapHeight];
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(mask, g_currMaskMapGpu, sizeof(uchar) * g_maskMapWidth * g_maskMapHeight, cudaMemcpyDeviceToHost));
    return mask;
}
void InstanceSegManager::setPersonMap(uchar* dynamicMap)
{
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(g_currPersonMaskGpu, dynamicMap, sizeof(uchar) * g_maskMapWidth * g_maskMapHeight, cudaMemcpyHostToDevice));
}
void InstanceSegManager::setObjMap(uchar* dynamicMap)
{
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(g_currObjMaskGpu, dynamicMap, sizeof(uchar) * g_maskMapWidth * g_maskMapHeight, cudaMemcpyHostToDevice));
}
uchar* InstanceSegManager::getPersonMapCpu()
{
    uchar* currPersonMask = new uchar[g_maskMapWidth * g_maskMapHeight];
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(currPersonMask, g_currPersonMaskGpu, sizeof(uchar) * g_maskMapWidth * g_maskMapHeight, cudaMemcpyDeviceToHost));
    return currPersonMask;
}
uchar* InstanceSegManager::getObjMapCpu()
{
    uchar* currObjMask = new uchar[g_maskMapWidth * g_maskMapHeight];
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(currObjMask, g_currObjMaskGpu, sizeof(uchar) * g_maskMapWidth * g_maskMapHeight, cudaMemcpyDeviceToHost));
    return currObjMask;
}
uchar* InstanceSegManager::getPersonMapGpu()
{
    return g_currPersonMaskGpu;
}
void InstanceSegManager::updateExistingDynamicPx_currFrame(const float* flowUGpu, const float* flowVGpu)
{
    //TODO

}