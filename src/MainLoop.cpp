
#include <MainLoop.h>
#include <TimingLog.h>

#include <CUDAImageManager.h>

#include <DualGPU.h>

#include <TrajectoryManager.h>
#include <GlobalAppState.h>
#include <DepthSensing/TimingLogDepthSensing.h>
#include <DepthSensing/Util.h>
#include <DepthSensing/CUDASceneRepHashSDF.h>
#include <DepthSensing/CUDARayCastSDF.h>
#include <DepthSensing/CUDAMarchingCubesHashSDF.h>
#include <DepthSensing/CUDAHistogramHashSDF.h>
#include <DepthSensing/CUDASceneRepChunkGrid.h>
#include <CUDAImageManager.h>
#include <iomanip>
#include <fstream>
#include <unistd.h>
#include "OpticalFlow/OpticalFlowManager.h"
#include "InstanceSeg/InstanceSegManager.h"
#include "VisualizationHelper.h"
#include "KMeans/Kmeans.h"
#include "KMeans/Point3.h"
#include "DynamicObjectManager.h"
#ifdef WITH_VISUALIZATION
#include <Output3DWrapper.h>
#include <PangolinOutputWrapper.h>
#endif
// Variables

ORB_SLAM3::System* SLAM = nullptr;
OpticalFlowManager* g_OpticalFlowManager = nullptr;
InstanceSegManager* g_InstanceSegManager = nullptr;
DynamicObjectManager* g_ObjectManager = nullptr;
CUDAImageManager* g_imageManager = nullptr;
#ifdef WITH_VISUALIZATION
Visualization::Output3DWrapper * wrapper = nullptr;
#endif

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
CUDASceneRepHashSDF*		g_sceneRep = NULL;
CUDARayCastSDF*				g_rayCast = NULL;
CUDAMarchingCubesHashSDF*	g_marchingCubesHashSDF = NULL;
CUDAHistrogramHashSDF*		g_historgram = NULL;
CUDASceneRepChunkGrid*		g_chunkGrid = NULL;

DepthCameraParams			g_depthCameraParams;

//managed externally
int surface_read_count = 0;
bool publish_rgb = true;
bool publish_depth = false;
bool publish_mesh = true;
bool publish_dynamic = true;

// Functions
/**
 * debug function
 * get rgbdSensor, current default select PrimeSenseSensor
 * */

void calculateSDFMap(float* sdfMap, const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation);
void removeExistingDynamicPx ( const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation);
void integrate ( const DepthCameraData& depthCameraData, const Eigen::Matrix4f & transformation);//uint* existingDynamicPx
void deIntegrate ( const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation);
void reintegrate(int currFrameNumber);

void StopScanningAndExtractIsoSurfaceMC ( const std::string& filename = "./scans/scan.ply", bool overwriteExistingFile = false );
void StopScanningAndExit ( bool aborted = false );

void ResetDepthSensing();

bool CreateDevice();
extern "C" void convertColorFloat4ToUCHAR4 ( uchar4* d_output, float4* d_input, unsigned int width, unsigned int height );

/*************************BundleFusion SDK Interface ********************/
bool initSystem (std::string vocFile, std::string settingFile, std::string app_config)
{
    GlobalAppState::getInstance().readMembers ( app_config );
    SLAM = new ORB_SLAM3::System(vocFile, settingFile, ORB_SLAM3::System::RGBD, true);
    Eigen::Matrix4f intrinsics = SLAM->GetDepthIntrinsics();
    std::cout<<intrinsics(0,0)<<" "<<intrinsics(0,1)<<" "<<intrinsics(0,2)<<" "<<intrinsics(0,3)<<std::endl;
    std::cout<<intrinsics(1,0)<<" "<<intrinsics(1,1)<<" "<<intrinsics(1,2)<<" "<<intrinsics(1,3)<<std::endl;
    std::cout<<intrinsics(2,0)<<" "<<intrinsics(2,1)<<" "<<intrinsics(2,2)<<" "<<intrinsics(2,3)<<std::endl;
    std::cout<<intrinsics(3,0)<<" "<<intrinsics(3,1)<<" "<<intrinsics(3,2)<<" "<<intrinsics(3,3)<<std::endl;
    float imageScale = SLAM->GetImageScale();
    uint imageWidth, imageHeight;
    SLAM->GetImageSize(imageWidth, imageHeight);
    try {
        g_InstanceSegManager = new InstanceSegManager(imageWidth, imageHeight);
        g_OpticalFlowManager = new OpticalFlowManager(imageWidth, imageHeight);
        g_ObjectManager = new DynamicObjectManager();
        g_imageManager = new CUDAImageManager ( GlobalAppState::get().s_integrationWidth, GlobalAppState::get().s_integrationHeight,
                                                imageWidth, imageHeight, intrinsics, imageScale, false );

#ifdef WITH_VISUALIZATION
        wrapper = new Visualization::PangolinOutputWrapper ( GlobalAppState::get().s_integrationWidth,GlobalAppState::get().s_integrationHeight );
#endif

        if ( !CreateDevice() )
        {
            std::cerr<<"Create Device failed. " << std::endl;
            return false;
        }
    }
    catch ( const std::exception& e )
    {
        //MessageBoxA(NULL, e.what(), "Exception caught", MB_ICONERROR);
        std::cerr<< ( "Exception caught" ) << std::endl;
        return false;
    }
    catch ( ... )
    {
        //MessageBoxA(NULL, "UNKNOWN EXCEPTION", "Exception caught", MB_ICONERROR);
        std::cerr<< ( "UNKNOWN EXCEPTION" ) << std::endl;;
        return false;
    }
    return true;
}

cv::Mat ucharToMat(const uchar4* p2, const int width, const int height)
{
    //cout<< "length: " << p2-> << endl;
    int img_width = width;
    int img_height = height;
    cv::Mat img(cv::Size(img_width, img_height), CV_8UC3);
    for (int i = 0; i < img_width * img_height; i++)
    {
        int b = p2[i].x;
        int g = p2[i].y;
        int r = p2[i].z;

        img.at<cv::Vec3b>(i / img_width, (i % img_width))[0] = r;
        img.at<cv::Vec3b>(i / img_width, (i % img_width))[1] = g;
        img.at<cv::Vec3b>(i / img_width, (i % img_width))[2] = b;


    }
    return img;
}

cv::Mat floatToMat(const float* p2, const int width, const int height)
{
    //cout<< "length: " << p2-> << endl;
    int img_width = width;
    int img_height = height;
    cv::Mat img(cv::Size(img_width, img_height), CV_32FC1);
    for (int i = 0; i < img_width * img_height; i++)
    {
        float d = p2[i];

        img.at<float>(i / img_width, (i % img_width)) = d;
    }
    return img;
}
//only reconstructInputRGBDFrame
bool reconstructInputRGBDFrame(cv::Mat& rgb, cv::Mat& depth, cv::Mat& personMask, cv::Mat& instanceMask, cv::Mat& objMask, Sophus::SE3f pose, std::vector<Eigen::Vector4f> objCloud, Eigen::Matrix4f objPose)
{
    cv::Mat depthClone = depth.clone();
    depthClone.convertTo(depthClone,CV_32F);
    cv::Mat depthFiltered = depthClone.clone();
    cv::bilateralFilter(depthClone, depthFiltered, 5, 1.0, 1.0);//, cv::BORDER_DEFAULT

    double depthFactor = SLAM->GetDepthFactor();
    bool bGotDepth = g_imageManager->process ( rgb, depthFiltered, depthFactor );
    uint currFrameNumber = g_imageManager->getCurrFrameNumber();
    uint width = g_imageManager->getIntegrationWidth();
    uint height = g_imageManager->getIntegrationHeight();
    if ( bGotDepth )
    {
        Eigen::Matrix4f transformCurrent = pose.matrix();
        std::cout<<"transform:"<<transformCurrent<<std::endl;
        for(int i = 0;i<objCloud.size();i++)
        {
            objCloud[i] = transformCurrent * objPose * objCloud[i];
        }
        if ( GlobalAppState::get().s_reconstructionEnabled )
        {

            uchar* personData = (uchar*)personMask.data;
            uchar* personGPU;
            MLIB_CUDA_SAFE_CALL(cudaMalloc(&personGPU, sizeof(uchar) * width * height));
            MLIB_CUDA_SAFE_CALL(cudaMemcpy(personGPU, personData, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));
            uchar* instanceData = (uchar*)instanceMask.data;
            uchar* instanceGPU;
            MLIB_CUDA_SAFE_CALL(cudaMalloc(&instanceGPU, sizeof(uchar) * width * height));
            MLIB_CUDA_SAFE_CALL(cudaMemcpy(instanceGPU, instanceData, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));

            if(currFrameNumber == 999){
                g_marchingCubesHashSDF->clearObjectSurfaceGPUNoChrunk(g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData(), 8);

            }
//            if(currFrameNumber != 74 && currFrameNumber != 415){
//                if(currFrameNumber > 999){
                    uchar* objData = (uchar*)objMask.data;
                    uchar* objGPU;
                    MLIB_CUDA_SAFE_CALL(cudaMalloc(&objGPU, sizeof(uchar) * width * height));
                    MLIB_CUDA_SAFE_CALL(cudaMemcpy(objGPU, objData, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));

                    DepthCameraData depthCameraData( g_imageManager->getIntegrateFrame ( currFrameNumber ).getDepthFrameGPU(), g_imageManager->getIntegrateFrame ( currFrameNumber ).getColorFrameGPU(),
                                                     instanceGPU, personGPU, objGPU);

                    integrate (depthCameraData, transformCurrent);
//                }
//                else{
//                    DepthCameraData depthCameraData( g_imageManager->getIntegrateFrame ( currFrameNumber ).getDepthFrameGPU(), g_imageManager->getIntegrateFrame ( currFrameNumber ).getColorFrameGPU(),
 //                                                    instanceGPU, personGPU);
//                    integrate (depthCameraData, transformCurrent);
//                }
            //}
        }
    }

#ifdef WITH_VISUALIZATION
    if ( wrapper != nullptr )
    {
        // raw color image
        const uchar4* d_color = g_imageManager->getLastIntegrateFrame().getColorFrameCPU();
        // raw depth image
        const float* d_depth = g_imageManager->getLastIntegrateFrame().getDepthFrameCPU();
        const float minDepth = GlobalAppState::get().s_sensorDepthMin;
        const float maxDepth = GlobalAppState::get().s_sensorDepthMax;

        // publish these data
        if ( publish_rgb )
        {
            wrapper->publishColorMap (d_color);
        }

        if ( publish_depth )
        {
            wrapper->publishDepthMap (d_depth);
        }
        if( currFrameNumber > 9999)
        {

            //g_DynamicManager->getCurrentNodes(vertices, vertex_num, bbox);

            wrapper->publishObjCloud(objCloud);

        }
        else{
            std::vector<Eigen::Vector4f> nn;
            wrapper->publishObjCloud(nn);
        }
        if ( publish_mesh )//
        {
            // surface get
            MarchingCubesData* march_cube = nullptr;

            if ( surface_read_count == 1 )
            {
                surface_read_count = 0;


                Timer t;

                march_cube = g_marchingCubesHashSDF->extractIsoSurfaceGPUNoChrunk ( g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData() );

                std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;
            }
            else
            {
                surface_read_count++;
            }

            if ( march_cube != nullptr )
            {
                wrapper->publishSurface ( march_cube );
            }
        }
    }
#endif
}
bool processInputRGBDFrame (cv::Mat& rgb, cv::Mat& depth, double tframe, vector<double> vTimestamps, vector<float>& vTimesTrack)
{
    cv::Mat depthClone = depth.clone();
    depthClone.convertTo(depthClone,CV_32F);
    cv::Mat depthFiltered = depthClone.clone();
    cv::bilateralFilter(depthClone, depthFiltered, 5, 1.0, 1.0);//, cv::BORDER_DEFAULT
    // Read Input
    ///////////////////////////////////////
    double depthFactor = SLAM->GetDepthFactor();
    bool bGotDepth = g_imageManager->process ( rgb, depthFiltered, depthFactor );
    uint currFrameNumber = g_imageManager->getCurrFrameNumber();
    SLAM->curr_number = currFrameNumber;
    uint width = g_imageManager->getIntegrationWidth();
    uint height = g_imageManager->getIntegrationHeight();
    Eigen::Matrix4f intrinsics = g_imageManager->getDepthIntrinsics();
    if (bGotDepth)
    {
//        uint nImages =vTimestamps.size();
//        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//        vTimesTrack[currFrameNumber]=ttrack;
//        double T=0;
//        if(currFrameNumber<nImages-1)
//            T = vTimestamps[currFrameNumber+1]-tframe;
//        else if(currFrameNumber>0)
//            T = tframe-vTimestamps[currFrameNumber-1];
//        if(ttrack<T)
//            usleep((T-ttrack)*1e6);
        if(g_InstanceSegManager)
        {
            cv::Mat imgMatTar = ucharToMat(g_imageManager->getIntegrateFrame ( currFrameNumber ).getColorRawCPU(), width, height);
            int* mask = NULL;
            g_InstanceSegManager->executeSegmentation(imgMatTar, &mask);
            cv::Mat instanceMat = cv::Mat(height, width, CV_8UC1);
            for (int i = 0; i < height; i++)
            {
                for(int j = 0;j < width;j++)
                {
                    if(mask[width * i + j] > 0)
                    {
                        instanceMat.at<uchar>(i, j) = mask[width * i + j];
                    }
                    else{
                        instanceMat.at<uchar>(i, j) = 0;
                    }
                }
            }
            cv::imwrite("../instance_mat/" + std::to_string(currFrameNumber) + ".png", instanceMat);


            int* maskGPU1;
            MLIB_CUDA_SAFE_CALL(cudaMalloc(&maskGPU1, sizeof(int) * width * height));
            MLIB_CUDA_SAFE_CALL(cudaMemcpy(maskGPU1, mask, sizeof(int) * width * height, cudaMemcpyHostToDevice));
            uchar* maskGPU;
            MLIB_CUDA_SAFE_CALL(cudaMalloc(&maskGPU, sizeof(uchar) * width * height));
            MLIB_CUDA_SAFE_CALL(cudaMemset(maskGPU, 0, sizeof(uchar) * width * height));
            CUDAImageUtil::int2uchar(maskGPU1, maskGPU, width, height);
            MLIB_CUDA_SAFE_CALL(cudaFree(maskGPU1));
            uchar* maskCPU = new uchar[width * height];
            MLIB_CUDA_SAFE_CALL(cudaMemcpy(maskCPU, maskGPU, sizeof(uchar) * width * height, cudaMemcpyDeviceToHost));
            MLIB_CUDA_SAFE_CALL(cudaFree(maskGPU));


            //uchar* segmentMask =(uchar*)segment_mat.data;
            g_InstanceSegManager->setMaskMap(maskCPU, width, height);
            //gen person mask
            g_InstanceSegManager->genPersonMask();

            uchar* personMask = g_InstanceSegManager->getPersonMapCpu();
            cv::Mat segment_mat = cv::Mat(height, width, CV_8UC1, personMask);
            cv::Mat se1 = cv::getStructuringElement(0, cv::Size(15,15));
            cv::Mat se2 = cv::getStructuringElement(0, cv::Size(2,2));
            //cv::erode(consistency_mat, consistency_mat, se1, cv::Point(-1, -1), 1);
            cv::dilate(segment_mat, segment_mat, se1, cv::Point(-1, -1), 1);
            //cv::floodFill(segment_mat, cv::Point(0,0), cv::Scalar(255));
            personMask = (uchar*)segment_mat.data;
            cv::imwrite("../dilate/" + std::to_string(currFrameNumber) + ".png", segment_mat);
            std::cout<<"width:"<<width<<", height:"<<height<<std::endl;

            cv::Mat personMat = cv::Mat(height, width, CV_8UC1);
            for (int i = 0; i < height; i++)
            {
                for(int j = 0;j < width;j++)
                {
                    if(personMask[width * i + j] > 0)
                    {
                        personMat.at<uchar>(i, j) = 255;
                    }
                    else{
                        personMat.at<uchar>(i, j) = 0;
                    }
                }
            }
            cv::imwrite("../person_mat/" + std::to_string(currFrameNumber) + ".png", personMat);
            cv::waitKey(1);

            uchar* instanceMask = g_InstanceSegManager->getInstanceMapCpu();
            cv::Mat objMat = cv::Mat(height, width, CV_8UC1);
            for (int i = 0; i < height; i++)
            {
                for(int j = 0;j < width;j++)
                {
                    if( instanceMask[width * i + j] ==8 )//(currFrameNumber > 151 && currFrameNumber < 3)
                    {
                        objMat.at<uchar>(i, j) = 0;
                    }
                    else{
                        objMat.at<uchar>(i, j) = 0;
                    }
                }
            }
            cv::dilate(objMat, objMat, se1, cv::Point(-1, -1), 1);

            cv::imwrite("../obj_mat/" + std::to_string(currFrameNumber) + ".png", objMat);
            cv::waitKey(1);

            g_InstanceSegManager->setPersonMap(personMask);
            g_InstanceSegManager->setObjMap(objMat.data);
        }
        //perform optical flow here and save optical flow in g_OpticalFlowManager.
        if(g_OpticalFlowManager && currFrameNumber > 0)
        {
            cv::Mat imgMatSrcLast1 = ucharToMat(g_imageManager->getIntegrateFrame( currFrameNumber - 1 ).getColorRawCPU(), width, height);
            cv::Mat imgMatTar = ucharToMat(g_imageManager->getIntegrateFrame ( currFrameNumber ).getColorRawCPU(), width, height);
            float* flowU = NULL;
            float* flowV = NULL;
            g_OpticalFlowManager->executeOpticalFlow(imgMatTar, imgMatSrcLast1, &flowU, &flowV);  //last<-current
            g_OpticalFlowManager->setOpticalFlowMap(flowU, flowV, width, height);
//            float* flowUGpu;
//            float* flowVGpu;
//            MLIB_CUDA_SAFE_CALL(cudaMalloc(&flowUGpu, sizeof(float) * width * height));
//            MLIB_CUDA_SAFE_CALL(cudaMalloc(&flowVGpu, sizeof(float) * width * height));
//            g_OpticalFlowManager->getOpticalFlowMapGpu(flowUGpu, flowVGpu);
//            //SE3 -> Matrix4x4
//            std::vector<Sophus::SE3f> trajectories = SLAM->ExtractTrajectoryTUM();
//            Eigen::Matrix4f intrinsics = g_imageManager->getDepthIntrinsics();
//            Eigen::Matrix4f matrix_last = trajectories[currFrameNumber - 1].matrix();
//            g_InstanceSegManager->updateExistingDynamicPx_currFrame(flowUGpu, flowVGpu);
//
//            MLIB_CUDA_SAFE_CALL(cudaFree(flowUGpu));
//            MLIB_CUDA_SAFE_CALL(cudaFree(flowVGpu));

        }
        SLAM->TrackRGBD(rgb, depthFiltered, g_InstanceSegManager->getPersonMapCpu(), g_InstanceSegManager->getObjMapCpu(), tframe);
        std::cout<<"tracking!"<<std::endl;
    }

    ///////////////////////////////////////
    // Fix old frames, and fuse the segment result to the TSDF model.
    ///////////////////////////////////////
    //printf("start reintegrate\n");
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
 //   reintegrate(currFrameNumber);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    ///////////////////////////////////////
    // Reconstruction of current frame
    ///////////////////////////////////////
   if ( bGotDepth )
   {

       std::vector<Sophus::SE3f> trajectories = SLAM->ExtractTrajectoryTUM();
       std::cout<<"trajectories:"<<trajectories.size() <<std::endl;
       if(trajectories.size() == 0)
       {
           return;
       }
       std::cout<< trajectories[trajectories.size() - 1].matrix()<<std::endl;
       Eigen::Matrix4f transformCurrent = trajectories[trajectories.size() - 1].matrix();
      ///////////////////////////////////////////////
      // track Dynamic node
      ///////////////////////////////////////////////
        if(g_OpticalFlowManager) {
            if (currFrameNumber > 0) {
                /////////////////////depSrc and depTar
                const float* depSrc_cpu = g_imageManager->getIntegrateFrame(currFrameNumber - 1).getDepthFrameCPU();
                const float* depTar_cpu = g_imageManager->getIntegrateFrame(currFrameNumber).getDepthFrameCPU();
                float* depSrc_gpu;
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&depSrc_gpu, sizeof(float) * width * height));
                MLIB_CUDA_SAFE_CALL(cudaMemcpy(depSrc_gpu, depSrc_cpu, sizeof(float) * width * height, cudaMemcpyHostToDevice));
                float* depTar_gpu;
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&depTar_gpu, sizeof(float) * width * height));
                MLIB_CUDA_SAFE_CALL(cudaMemcpy(depTar_gpu, depTar_cpu, sizeof(float) * width * height, cudaMemcpyHostToDevice));
                /////////////////////colorSrc and colorTar
                const uchar4 * colorSrc_cpu = g_imageManager->getIntegrateFrame(currFrameNumber -1).getColorRawCPU();
                const uchar4 * colorTar_cpu = g_imageManager->getIntegrateFrame(currFrameNumber).getColorRawCPU();
                const uchar4* colorSrc_gpu = g_imageManager->getIntegrateFrame(currFrameNumber - 1).getColorFrameGPU();
                const uchar4* colorTar_gpu = g_imageManager->getIntegrateFrame(currFrameNumber).getColorFrameGPU();
                /////////////////////init transform of last frame

                Eigen::Matrix4f transformLast = trajectories[trajectories.size() - 1].matrix();
                //Eigen::Matrix4f transformCanonical = trajectories[19].matrix();
                /////////////////////optical flow
                float *flowUGpu;
                float *flowVGpu;
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&flowUGpu, sizeof(float)*width*height));
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&flowVGpu, sizeof(float)*width*height));
                g_OpticalFlowManager->getOpticalFlowMapGpu(flowUGpu, flowVGpu);
                /////////////////////////////compute 3d points and normals in current frame////////////////////////
                if(g_ObjectManager)
                {
                    float *flowUCpu = new float[width * height];
                    MLIB_CUDA_SAFE_CALL(cudaMemcpy(flowUCpu, flowUGpu, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
                    float *flowVCpu = new float[width * height];
                    MLIB_CUDA_SAFE_CALL(cudaMemcpy(flowVCpu, flowVGpu,  sizeof(float) * width * height, cudaMemcpyDeviceToHost));
                    if (g_ObjectManager->getObjects_cloud_num()!=0 )
                    {
                        g_ObjectManager->trackObjects(colorSrc_cpu, colorTar_cpu, depSrc_cpu, depTar_cpu, flowUCpu, flowVCpu, intrinsics, currFrameNumber, transformLast, transformCurrent, g_InstanceSegManager->getPersonMapCpu(), g_InstanceSegManager->getInstanceMapCpu(), width, height);

                    }

                    //g_ObjectManager->updateDynamicPx();
                }
                /////////////////////extend dynamic
                //uint *existingDynamicPx;
                //MLIB_CUDA_SAFE_CALL(cudaMalloc(&existingDynamicPx, sizeof(uint) * width * height));
                float* consistency_float_px_cpu = new float[width * height];
                uchar* consistency_gpu = g_OpticalFlowManager->extractMotionConsistency(consistency_float_px_cpu, CONSISTENCY_THRESHOLD,
                                                                                        g_InstanceSegManager->getPersonMapGpu(), depTar_gpu, depSrc_gpu, colorTar_gpu, colorSrc_gpu,
                                                                                        flowUGpu, flowVGpu, width, height, intrinsics, transformLast, transformCurrent, currFrameNumber);
                uchar* consistency_cpu = new uchar[width * height];
                MLIB_CUDA_SAFE_CALL(cudaMemcpy(consistency_cpu, consistency_gpu, sizeof(uchar) * width * height, cudaMemcpyDeviceToHost));
                MLIB_CUDA_SAFE_CALL(cudaFree(consistency_gpu));
                cv::Mat consistency_mat = cv::Mat(height, width, CV_8UC1, consistency_cpu);
                cv::Mat se1 = cv::getStructuringElement(0, cv::Size(5,5));
                cv::Mat se2 = cv::getStructuringElement(0, cv::Size(2,2));
                cv::erode(consistency_mat, consistency_mat, se1, cv::Point(-1, -1), 1);
                cv::dilate(consistency_mat, consistency_mat, se1, cv::Point(-1, -1), 1);

                cv::dilate(consistency_mat, consistency_mat, se1, cv::Point(-1, -1), 1);
                cv::erode(consistency_mat, consistency_mat, se1, cv::Point(-1, -1), 1);
                ///////////////////////////////////////////////////////////////////////////////////////////////////
                cv::imwrite("../consistency_mat/" + std::to_string(currFrameNumber) + ".png", consistency_mat);
                std::vector<int> biggestIndexes;
                cv::Mat labels, stats, centroids;
                int connectedNum = cv::connectedComponentsWithStats(consistency_mat, labels, stats, centroids);
                uint * connection =(uint*)labels.data;
                uint* connection_gpu;
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&connection_gpu, sizeof(uint) * width * height));
                MLIB_CUDA_SAFE_CALL(cudaMemcpy(connection_gpu, connection, sizeof(uint) * width * height,cudaMemcpyHostToDevice));
                bool genDynamicObj = false;
                if(currFrameNumber == 166 && genDynamicObj)
                {
                    uchar objCat = 8;
                    MarchingCubesData* dyna_march_cube = nullptr;
                    dyna_march_cube = g_marchingCubesHashSDF->extractObjectSurfaceGPUNoChrunk ( g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData(), objCat );
                    uint* size = new uint ( 0 );
                    MLIB_CUDA_SAFE_CALL(cudaMemcpy ( size, dyna_march_cube->d_numTriangles, sizeof ( uint ), cudaMemcpyDeviceToHost ) );
                    int size_x64 = int ( *size );
                    float3* triangles_gpu = (float3*)(dyna_march_cube->d_triangles);
                    float3* triangles_cpu = new float3[size_x64 * 6];
                    MLIB_CUDA_SAFE_CALL(cudaMemcpy(triangles_cpu, triangles_gpu, sizeof(float3) * size_x64 * 6, cudaMemcpyDeviceToHost));

//                    //KMeans
//                    std::vector<kmeans::Point3<float>> pointCloud;
//                    for (int i = 0;i<size_x64;i++) {
//                        if (!std::isinf(triangles_cpu[6*i+0].x)&&!std::isinf(triangles_cpu[6*i+0].y)&&!std::isinf(triangles_cpu[6*i+0].z))
//                            pointCloud.emplace_back(triangles_cpu[6*i+0].x,triangles_cpu[6*i+0].y,triangles_cpu[6*i+0].z);
//                        if (!std::isinf(triangles_cpu[6*i+2].x)&&!std::isinf(triangles_cpu[6*i+2].y)&&!std::isinf(triangles_cpu[6*i+2].z))
//                            pointCloud.emplace_back(triangles_cpu[6*i+2].x,triangles_cpu[6*i+2].y,triangles_cpu[6*i+2].z);
//                        if (!std::isinf(triangles_cpu[6*i+4].x)&&!std::isinf(triangles_cpu[6*i+4].y)&&!std::isinf(triangles_cpu[6*i+4].z))
//                            pointCloud.emplace_back(triangles_cpu[6*i+4].x,triangles_cpu[6*i+4].y,triangles_cpu[6*i+4].z);
//                    }
//                    kmeans::Kmeans<float> k_mean(3,pointCloud);
//                    std::vector<std::vector<size_t>> cluster = k_mean.GetClusterPoint();
                    //
//                    int cluster_point_num =0;
//                    int cluster_idx=0;
//                    for (int i=0; i<cluster.size(); i++)
//                    {
//                        if (cluster_point_num < cluster[i].size())
//                        {
//                            cluster_idx=i;
//                            cluster_point_num = cluster[i].size();
//                        }
//                    }
                    std::vector<Eigen::Vector4f> objCloud;

                    string strLine2;
                    ifstream inFile2("../initObjCloud.txt");
                    while(getline(inFile2, strLine2)) // line中不包括每行的换行符
                    {
                        istringstream ss(strLine2);
                        Eigen::Vector4f objPoint = Eigen::Vector4f(0.0,0.0,0.0,1.0);
                        std::vector<float> words;
                        for(int i = 0;i<3;i++)
                        {
                            float word;
                            ss >> word;
                            words.push_back(word);
                        }
                        objPoint.x() = words[0];
                        objPoint.y() = words[1];
                        objPoint.z() = words[2];
                        if(objPoint.z() < 2.5)
                            objCloud.push_back(objPoint);
                    }

//                    for (int i=0; i<size_x64; i++)
//                    {
//                        if (!std::isinf(triangles_cpu[6*i+0].x)&&!std::isinf(triangles_cpu[6*i+0].y)&&!std::isinf(triangles_cpu[6*i+0].z))
//                            objCloud.push_back(transformCurrent.inverse() * Eigen::Vector4f(triangles_cpu[6*i+0].x,triangles_cpu[6*i+0].y,triangles_cpu[6*i+0].z, 1.0));
//                        if (!std::isinf(triangles_cpu[6*i+2].x)&&!std::isinf(triangles_cpu[6*i+2].y)&&!std::isinf(triangles_cpu[6*i+2].z))
//                            objCloud.push_back(transformCurrent.inverse() * Eigen::Vector4f(triangles_cpu[6*i+2].x,triangles_cpu[6*i+2].y,triangles_cpu[6*i+2].z, 1.0));
//                        if (!std::isinf(triangles_cpu[6*i+4].x)&&!std::isinf(triangles_cpu[6*i+4].y)&&!std::isinf(triangles_cpu[6*i+4].z))
//                            objCloud.push_back(transformCurrent.inverse() * Eigen::Vector4f(triangles_cpu[6*i+4].x,triangles_cpu[6*i+4].y,triangles_cpu[6*i+4].z, 1.0));
//                    }
                    g_ObjectManager->genObjects(objCloud, objCat);

//                    ofstream ofs;
//                    for (int i = 0; i < cluster.size(); i++)
//                    {
//                        ofs.open("cluster" + std::to_string(i + 1) + ".txt");
//                        for (size_t j = 0; j < cluster[i].size(); j++)
//                        {
//                            ofs << pointCloud[cluster[i][j]].x << " "
//                                   << pointCloud[cluster[i][j]].y << " "
//                                   << pointCloud[cluster[i][j]].z << " " << std::endl;
//                        }
//                        ofs.close();
//                    }
                    ofstream ofs;
                    ofs.open("../initObjCloud.txt",ios::out );
                    for (int i = 0;i<objCloud.size();i++) {
                        if (!std::isinf(objCloud[i].x())&&!std::isinf(objCloud[i].y())&&!std::isinf(objCloud[i].z()))
                            ofs << objCloud[i].x() << " " << objCloud[i].y() << " " << objCloud[i].z() << endl;
                    }
                    ofs.close();
                    g_marchingCubesHashSDF->clearObjectSurfaceGPUNoChrunk(g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData(), 8);
                }


                MLIB_CUDA_SAFE_CALL(cudaFree(depSrc_gpu));
                MLIB_CUDA_SAFE_CALL(cudaFree(depTar_gpu));
                //MLIB_CUDA_SAFE_CALL(cudaFree(existingDynamicPx));
                MLIB_CUDA_SAFE_CALL(cudaFree(connection_gpu));
                MLIB_CUDA_SAFE_CALL(cudaFree(flowUGpu));
                MLIB_CUDA_SAFE_CALL(cudaFree(flowVGpu));
            }
        }
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        if ( GlobalAppState::get().s_reconstructionEnabled)
        {

            DepthCameraData depthCameraData( g_imageManager->getIntegrateFrame ( currFrameNumber ).getDepthFrameGPU(), g_imageManager->getIntegrateFrame ( currFrameNumber ).getColorFrameGPU(),
                                             g_InstanceSegManager->getInstanceMapGpu(), g_InstanceSegManager->getPersonMapGpu());
            integrate (depthCameraData, transformCurrent);
        }
       std::cout<<"eee"<<std::endl;
        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>> ( t2 - t1 +t4 - t3 );
        std::cout << "depthSensing time cost = " << time_used.count() << " seconds." << std::endl;

        std::chrono::duration<double> time_total = std::chrono::duration_cast<std::chrono::duration<double>> ( t4 - t1 );
        std::cout << "total time cost = " << time_total.count() << " seconds." << std::endl;

    }

#ifdef WITH_VISUALIZATION
    if ( wrapper != nullptr )
    {
        // raw color image
        const uchar4* d_color = g_imageManager->getLastIntegrateFrame().getColorFrameCPU();
        // raw depth image
        const float* d_depth = g_imageManager->getLastIntegrateFrame().getDepthFrameCPU();
        const float minDepth = GlobalAppState::get().s_sensorDepthMin;
        const float maxDepth = GlobalAppState::get().s_sensorDepthMax;

        // publish these data
        if ( publish_rgb )
        {
            /*uint width = g_imageManager->getIntegrationWidth();
            uint height = g_imageManager->getIntegrationHeight();
            cv::Mat colorMat = cv::Mat(height, width, CV_8UC3);
            for (int i = 0; i < height; i++)
            {
                for(int j = 0;j < width;j++)
                {
                    colorMat.at<Vec3b>(i, j)[0] = d_color[i * width + j].x;
                    colorMat.at<Vec3b>(i, j)[1] = d_color[i * width + j].y;
                    colorMat.at<Vec3b>(i, j)[2] = d_color[i * width + j].z;
                }

            }
            cv::imshow("22222", colorMat);
            cv::waitKey(1);
            uchar4* newColor = new uchar4[width * height];
            for (int i = 0; i < height; i++)
            {
                for(int j = 0;j < width;j++)
                {
                    newColor[i * width + j].x = colorMat.at<Vec3b>(i, j)[0];
                    newColor[i * width + j].y = colorMat.at<Vec3b>(i, j)[1];
                    newColor[i * width + j].z = colorMat.at<Vec3b>(i, j)[2];
                    newColor[i * width + j].w = 0;
                }

            }*/
            wrapper->publishColorMap (d_color);
        }

        if ( publish_depth )
        {
            wrapper->publishDepthMap (d_depth);
        }
        if( publish_dynamic )
        {
            vector<float3x2*> vertices;
            vector<uint> vertex_num;

            vector<float3*> bbox;
            //g_DynamicManager->getCurrentNodes(vertices, vertex_num, bbox);

            wrapper->publishDynamic(vertices, vertex_num, bbox);

        }
        if ( publish_mesh )//
        {
            // surface get
            MarchingCubesData* march_cube = nullptr;

            if ( surface_read_count == 1 )
            {
                surface_read_count = 0;


                Timer t;

                march_cube = g_marchingCubesHashSDF->extractIsoSurfaceGPUNoChrunk ( g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData() );

                std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;
            }
            else
            {
                surface_read_count++;
            }

            if ( march_cube != nullptr )
            {
                wrapper->publishSurface ( march_cube );
            }

//            std::vector<Sophus::SE3f> trajectories = SLAM->ExtractTrajectoryTUM();
//            float* trajs_float = new float[3 * trajectories.size()];
//            float* pose = new float[16];
//            for ( size_t i = 0; i < trajectories.size(); ++i )
//            {
//                Eigen::Matrix4f transform = trajectories[i].matrix();
//                trajs_float[3*i + 0] = transform(0,3);
//                trajs_float[3*i + 1] = transform(1,3);
//                trajs_float[3*i + 2] = transform(2,3);
//                if ( i == trajectories.size()-1 )
//                {
//                    mat4f& last_pose = transform;
//                    last_pose.transpose();
//                    for ( size_t j = 0; j < 16; ++j )
//                        pose[j] = last_pose[j];
//                }
//
//            }

            //wrapper->publishAllTrajetory ( trajs_float, trajectory.size() );


            //wrapper->publishCurrentCameraPose ( pose );
//
            //delete pose;

            //delete trajs_float;

        }

    }
#endif

 return true;
}

void setPublishRGBFlag ( bool publish_flag )
{
    publish_rgb = publish_flag;
}

void setPublishMeshFlag ( bool publish_flag )
{
    publish_mesh = publish_flag;
}

bool saveMeshIntoFile ( const std::string& filename, bool overwriteExistingFile /*= false*/ )
{
    //g_sceneRep->debugHash();
    //g_chunkGrid->debugCheckForDuplicates();

    std::cout << "running marching cubes...1" << std::endl;

    Timer t;


    g_marchingCubesHashSDF->clearMeshBuffer();
    if ( !GlobalAppState::get().s_streamingEnabled )
    {
        //g_chunkGrid->stopMultiThreading();
        //g_chunkGrid->streamInToGPUAll();
        g_marchingCubesHashSDF->extractIsoSurface ( g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData() );
        //g_chunkGrid->startMultiThreading();
    }
    else
    {
        vec4f posWorld = vec4f ( GlobalAppState::get().s_streamingPos, 1.0f ); // trans lags one frame
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );
        g_marchingCubesHashSDF->extractIsoSurface ( *g_chunkGrid, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius );
    }

    const mat4f& rigidTransform = mat4f::identity();//g_lastRigidTransform
    g_marchingCubesHashSDF->saveMesh ( filename, &rigidTransform, overwriteExistingFile );

    std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;

    return true;

    //g_sceneRep->debugHash();
    //g_chunkGrid->debugCheckForDuplicates();
}



bool deinitSystem()
{

    SAFE_DELETE ( g_sceneRep );
    SAFE_DELETE ( g_rayCast );
    SAFE_DELETE ( g_marchingCubesHashSDF );
    SAFE_DELETE ( g_historgram );
    SAFE_DELETE ( g_chunkGrid );


    SAFE_DELETE ( g_imageManager );


#ifdef WITH_VISUALIZATION
    if ( wrapper != nullptr )
    {
        wrapper->noticeFinishFlag();
    }
#endif

    SLAM->Shutdown();
//
//    // Tracking time statistics
////    sort(vTimesTrack.begin(),vTimesTrack.end());
////    float totaltime = 0;
////    for(int ni=0; ni<nImages; ni++)
////    {
////        totaltime+=vTimesTrack[ni];
////    }
////    cout << "-------" << endl << endl;
////    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
////    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM->SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return true;

}

/************************************************************************/



bool CreateDevice()
{

    g_sceneRep = new CUDASceneRepHashSDF ( CUDASceneRepHashSDF::parametersFromGlobalAppState ( GlobalAppState::get() ) );
    //g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppState(GlobalAppState::get(), g_imageManager->getColorIntrinsics(), g_CudaImageManager->getColorIntrinsicsInv()));
    g_rayCast = new CUDARayCastSDF ( CUDARayCastSDF::parametersFromGlobalAppState ( GlobalAppState::get(), g_imageManager->getDepthIntrinsics(), g_imageManager->getDepthIntrinsicsInv() ) );

    g_marchingCubesHashSDF = new CUDAMarchingCubesHashSDF ( CUDAMarchingCubesHashSDF::parametersFromGlobalAppState ( GlobalAppState::get() ) );
    g_historgram = new CUDAHistrogramHashSDF ( g_sceneRep->getHashParams() );

    if ( GlobalAppState::get().s_streamingEnabled )
    {
        g_chunkGrid = new CUDASceneRepChunkGrid ( g_sceneRep,
                GlobalAppState::get().s_streamingVoxelExtents,
                GlobalAppState::get().s_streamingGridDimensions,
                GlobalAppState::get().s_streamingMinGridPos,
                GlobalAppState::get().s_streamingInitialChunkListSize,
                GlobalAppState::get().s_streamingEnabled,
                GlobalAppState::get().s_streamingOutParts );
    }
    if ( !GlobalAppState::get().s_reconstructionEnabled )
    {
        GlobalAppState::get().s_RenderMode = 2;
    }

    g_depthCameraParams.fx = g_imageManager->getDepthIntrinsics() ( 0, 0 ); //TODO check intrinsics
    g_depthCameraParams.fy = g_imageManager->getDepthIntrinsics() ( 1, 1 );

    g_depthCameraParams.mx = g_imageManager->getDepthIntrinsics() ( 0, 2 );
    g_depthCameraParams.my = g_imageManager->getDepthIntrinsics() ( 1, 2 );
    g_depthCameraParams.m_sensorDepthWorldMin = GlobalAppState::get().s_renderDepthMin;
    g_depthCameraParams.m_sensorDepthWorldMax = GlobalAppState::get().s_renderDepthMax;
    g_depthCameraParams.m_imageWidth = g_imageManager->getIntegrationWidth();
    g_depthCameraParams.m_imageHeight = g_imageManager->getIntegrationHeight();
    std::cout<<g_depthCameraParams.fx << "," << g_depthCameraParams.fy << "," <<g_depthCameraParams.mx << "," <<g_depthCameraParams.my << "," <<g_depthCameraParams.m_sensorDepthWorldMin << "," <<g_depthCameraParams.m_sensorDepthWorldMax << "," <<g_depthCameraParams.m_imageWidth << "," <<g_depthCameraParams.m_imageHeight << std::endl;
    DepthCameraData::updateParams ( g_depthCameraParams );

    //std::vector<DXGI_FORMAT> rtfFormat;
    //rtfFormat.push_back(DXGI_FORMAT_R8G8B8A8_UNORM); // _SRGB
    //V_RETURN(g_RenderToFileTarget.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_rayCastWidth, GlobalAppState::get().s_rayCastHeight, rtfFormat));

    //g_CudaImageManager->OnD3D11CreateDevice(pd3dDevice);

    return true;
}
/*void insertMask(const DepthCameraData& depthCameraData, const mat4f& transformation )
{
    if ( GlobalAppState::get().s_integrationEnabled )
    {
        unsigned int* d_bitMask = NULL;
        if ( g_chunkGrid ) d_bitMask = g_chunkGrid->getBitMaskGPU();
        g_sceneRep->insertMask ( transformation, depthCameraData, g_depthCameraParams, d_bitMask );
    }
}*/




void removeExistingDynamicPx( const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation)
{
    if ( GlobalAppState::get().s_streamingEnabled )
    {
        Eigen::Vector4f trans = transformation * Eigen::Vector4f(GlobalAppState::getInstance().s_streamingPos.x,
                                                                 GlobalAppState::getInstance().s_streamingPos.y,
                                                                 GlobalAppState::getInstance().s_streamingPos.z, 1.0);
        vec4f posWorld = vec4f ( trans(0), trans(1), trans(2), trans(3) ); // trans laggs one frame *trans
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );

        g_chunkGrid->streamOutToCPUPass0GPU ( p, GlobalAppState::get().s_streamingRadius, true, true );
        g_chunkGrid->streamInToGPUPass1GPU ( true );
    }
    if ( GlobalAppState::get().s_integrationEnabled )
    {
        unsigned int* d_bitMask = NULL;
        if ( g_chunkGrid )
            d_bitMask = g_chunkGrid->getBitMaskGPU();
        g_sceneRep->removeExistingDynamicPx ( transformation, depthCameraData, g_depthCameraParams, d_bitMask);


    }
}
void calculateSDFMap(float* sdfMap, const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation)
{
    if ( GlobalAppState::get().s_streamingEnabled )
    {
        Eigen::Vector4f trans = transformation * Eigen::Vector4f(GlobalAppState::getInstance().s_streamingPos.x,
                                                                 GlobalAppState::getInstance().s_streamingPos.y,
                                                                 GlobalAppState::getInstance().s_streamingPos.z, 1.0);
        vec4f posWorld = vec4f ( trans(0), trans(1), trans(2), trans(3) ); // trans laggs one frame *trans
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );

        g_chunkGrid->streamOutToCPUPass0GPU ( p, GlobalAppState::get().s_streamingRadius, true, true );
        g_chunkGrid->streamInToGPUPass1GPU ( true );
    }
    if ( GlobalAppState::get().s_integrationEnabled )
    {
        unsigned int* d_bitMask = NULL;
        if ( g_chunkGrid )
            d_bitMask = g_chunkGrid->getBitMaskGPU();
        g_sceneRep->calculateSDFMap ( sdfMap, transformation, depthCameraData, g_depthCameraParams, d_bitMask);


    }
}
void integrate ( const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation)//uint* existingDynamicPx
{
    if ( GlobalAppState::get().s_streamingEnabled )
    {
        Eigen::Vector4f trans_posWorld = transformation * Eigen::Vector4f(GlobalAppState::getInstance().s_streamingPos.x,
                                                                            GlobalAppState::getInstance().s_streamingPos.y,
                                                                            GlobalAppState::getInstance().s_streamingPos.z, 1.0f);
        vec4f posWorld = vec4f ( trans_posWorld(0), trans_posWorld(1), trans_posWorld(2), trans_posWorld(3) ); // trans laggs one frame *trans
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );

        g_chunkGrid->streamOutToCPUPass0GPU ( p, GlobalAppState::get().s_streamingRadius, true, true );
        g_chunkGrid->streamInToGPUPass1GPU ( true );
    }

    if ( GlobalAppState::get().s_integrationEnabled )
    {
        unsigned int* d_bitMask = NULL;
        if ( g_chunkGrid ) d_bitMask = g_chunkGrid->getBitMaskGPU();

        g_sceneRep->integrate ( transformation, depthCameraData, g_depthCameraParams, d_bitMask);


    }
    //else {
    //	//compactification is required for the ray cast splatting
    //	g_sceneRep->setLastRigidTransformAndCompactify(transformation);	//TODO check this
    //}
}

void deIntegrate ( const DepthCameraData& depthCameraData, const Eigen::Matrix4f& transformation)
{
    if ( GlobalAppState::get().s_streamingEnabled )
    {
        Eigen::Vector4f trans_posWorld = transformation * Eigen::Vector4f(GlobalAppState::getInstance().s_streamingPos.x,
                                                                          GlobalAppState::getInstance().s_streamingPos.y,
                                                                          GlobalAppState::getInstance().s_streamingPos.z, 1.0f);
        vec4f posWorld = vec4f ( trans_posWorld(0), trans_posWorld(1), trans_posWorld(2), trans_posWorld(3) ); // trans laggs one frame *trans
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );

        g_chunkGrid->streamOutToCPUPass0GPU ( p, GlobalAppState::get().s_streamingRadius, true, true );
        g_chunkGrid->streamInToGPUPass1GPU ( true );
    }

    if ( GlobalAppState::get().s_integrationEnabled )
    {
        unsigned int* d_bitMask = NULL;
        if ( g_chunkGrid ) d_bitMask = g_chunkGrid->getBitMaskGPU();
        g_sceneRep->deIntegrate ( transformation, depthCameraData, g_depthCameraParams, d_bitMask);
    }
    //else {
    //	//compactification is required for the ray cast splatting
    //	g_sceneRep->setLastRigidTransformAndCompactify(transformation);	//TODO check this
    //}
}



void reintegrate( int currFrameNumber)
{
    //find the 10(maxPerFrameFixes) frames with the biggest transform, and the frames with the sementic information
//    const unsigned int maxPerFrameFixes = GlobalAppState::get().s_maxFrameFixes;
//    TrajectoryManager* tm = g_bundler->getTrajectoryManager();
//    //std::cout << "reintegrate():" << tm->getNumActiveOperations() << " : " << tm->getNumOptimizedFrames() << std::endl;
//    uint width = g_imageManager->getIntegrationWidth();
//    uint height = g_imageManager->getIntegrationHeight();
//    if ( tm->getNumActiveOperations() < maxPerFrameFixes)
//    {
//        tm->generateUpdateLists();
//        //if (GlobalBundlingState::get().s_verbose) {
//        //	if (tm->getNumActiveOperations() == 0)
//        //		std::cout << __FUNCTION__ << " :  no more work (everything is reintegrated)" << std::endl;
//        //}
//    }
//
//    for ( unsigned int fixes = 0; fixes < maxPerFrameFixes; fixes++ )
//    {
//        mat4f newTransform = mat4f::zero();
//        mat4f oldTransform = mat4f::zero();
//        unsigned int frameIdx = ( unsigned int )-1;
//
//
//        if ( tm->getTopFromDeIntegrateList ( oldTransform, frameIdx ) )
//        {
//
////          DepthCameraData depthCameraData( f.getDepthFrameGPU(), f.getColorFrameGPU(), f.getDynamicMaskFrameGPU() );
//            MLIB_ASSERT ( !isnan ( oldTransform[0] ) && !isnan ( newTransform[0] ) && oldTransform[0] != -std::numeric_limits<float>::infinity() && newTransform[0] != -std::numeric_limits<float>::infinity() );
//
//            //deIntegrate ( depthCameraData, oldTransform);
//            MLIB_CUDA_SAFE_CALL(cudaFree(bbox_gpu));
//
//            continue;
//        }
//        else if ( tm->getTopFromIntegrateList ( newTransform, frameIdx ) )
//        {
//            auto& f = g_imageManager->getIntegrateFrame ( frameIdx );
//            MLIB_ASSERT ( !isnan ( newTransform[0] ) && newTransform[0] != -std::numeric_limits<float>::infinity() );
//
//            //integrate ( depthCameraDataErode, newTransform);
//            tm->confirmIntegration ( frameIdx );
//            continue;
//        }
//        else if ( tm->getTopFromReIntegrateList ( oldTransform, newTransform, frameIdx ) )
//        {
//            auto& f = g_imageManager->getIntegrateFrame ( frameIdx );
//            DepthCameraData depthCameraData( f.getDepthFrameGPU(), f.getColorFrameGPU(), f.getDynamicBoxFrameGPU() );
//            DepthCameraData depthCameraDataErode( f.getDepthFrameGPU(), f.getColorFrameGPU(), f.getDynamicBoxErodeFrameGPU() );
//            //uint *DynamicBoxFrameCpu = new  uint[width * height];
//            //MLIB_CUDA_SAFE_CALL(cudaMemcpy(DynamicBoxFrameCpu,f.getDynamicBoxFrameGPU(),sizeof(uint)*width * height, cudaMemcpyDeviceToHost));
//            //VisualizationHelper::ShowUint(DynamicBoxFrameCpu, width, height,"/home/user/2022AI/Large-scale_dynamic_scene_reconstruction/DynamicBoxFrame/" + std::to_string(currFrameNumber)+"---"+std::to_string(frameIdx));
//            //uint *DynamicBoxErodeFrameCpu = new  uint[width * height];
//            //MLIB_CUDA_SAFE_CALL(cudaMemcpy(DynamicBoxErodeFrameCpu,f.getDynamicBoxErodeFrameGPU(),sizeof(uint)*width * height, cudaMemcpyDeviceToHost));
//            //VisualizationHelper::ShowUint(DynamicBoxErodeFrameCpu, width, height,"/home/user/2022AI/Large-scale_dynamic_scene_reconstruction/DynamicBoxErodeFrame/"+ std::to_string(currFrameNumber)+"---"+std::to_string(frameIdx));
//            MLIB_ASSERT ( !isnan ( oldTransform[0] ) && !isnan ( newTransform[0] ) && oldTransform[0] != -std::numeric_limits<float>::infinity() && newTransform[0] != -std::numeric_limits<float>::infinity() );
//
//            deIntegrate ( depthCameraData, oldTransform);
//            integrate ( depthCameraDataErode, newTransform);
//            tm->confirmIntegration ( frameIdx );
//            continue;
//        }
//        else
//        {
//            break; //no more work to do
//        }
//    }
//    g_sceneRep->garbageCollect();
}

void StopScanningAndExtractIsoSurfaceMC ( const std::string& filename, bool overwriteExistingFile /*= false*/ )
{

    std::cout << "running marching cubes...1" << std::endl;

    Timer t;


    g_marchingCubesHashSDF->clearMeshBuffer();
    if ( !GlobalAppState::get().s_streamingEnabled )
    {
        //g_chunkGrid->stopMultiThreading();
        //g_chunkGrid->streamInToGPUAll();
        g_marchingCubesHashSDF->extractIsoSurface ( g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData() );
        //g_chunkGrid->startMultiThreading();
    }
    else
    {
        vec4f posWorld = vec4f ( GlobalAppState::get().s_streamingPos, 1.0f ); // trans lags one frame
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );
        g_marchingCubesHashSDF->extractIsoSurface ( *g_chunkGrid, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius );
    }

    const mat4f& rigidTransform = mat4f::identity();//g_lastRigidTransform
    g_marchingCubesHashSDF->saveMesh ( filename, &rigidTransform, overwriteExistingFile );

    std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;

    //g_sceneRep->debugHash();
    //g_chunkGrid->debugCheckForDuplicates();
}

void ResetDepthSensing()
{
    g_sceneRep->reset();
    //g_Camera.Reset();
    if ( g_chunkGrid )
    {
        g_chunkGrid->reset();
    }
}


