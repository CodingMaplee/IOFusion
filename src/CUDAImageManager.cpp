#include "stdafx.h"

#include "CUDAImageManager.h"

bool		CUDAImageManager::ManagedRGBDInputFrame::s_bIsOnGPU = false;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::s_width = 0;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::s_height = 0;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::s_widthRaw = 0;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::s_heightRaw = 0;

float*		CUDAImageManager::ManagedRGBDInputFrame::s_depthIntegrationGlobal = NULL;
uchar4*		CUDAImageManager::ManagedRGBDInputFrame::s_colorIntegrationGlobal = NULL;
float*		CUDAImageManager::ManagedRGBDInputFrame::s_depthRawGlobal = NULL;
uchar4*		CUDAImageManager::ManagedRGBDInputFrame::s_colorRawGlobal = NULL;

CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeColorGPU = NULL;
CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeDepthGPU = NULL;

CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeColorCPU = NULL;
CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeDepthCPU = NULL;




bool CUDAImageManager::process ( cv::Mat& rgb, cv::Mat& depth, double depthFactor )
{
    m_data.push_back ( ManagedRGBDInputFrame() );
    ManagedRGBDInputFrame& frame = m_data.back();
    frame.alloc();
    uint initWidth = rgb.cols;
    uint initHeight = rgb.rows;

    const unsigned int bufferDimColorInput = initWidth * initHeight;
    float* depthInput = new float[bufferDimColorInput];
    vec4uc* colorInput = new vec4uc[bufferDimColorInput];

    readDepthAndColor(rgb, depth, depthInput, colorInput, depthFactor);

    MLIB_CUDA_SAFE_CALL ( cudaMemcpy ( d_colorInput, colorInput, sizeof ( uchar4 ) * bufferDimColorInput, cudaMemcpyHostToDevice ) );
    if ( ManagedRGBDInputFrame::s_bIsOnGPU )
    {
        CUDAImageUtil::copy<uchar4> ( frame.m_colorRaw, d_colorInput, initWidth, initHeight );
    }
    else
    {
        memcpy ( frame.m_colorRaw, colorInput, sizeof ( uchar4 ) *bufferDimColorInput );

    }


    if ((initWidth == m_widthIntegration ) && (initHeight == m_heightIntegration ) )
    {
        if ( ManagedRGBDInputFrame::s_bIsOnGPU )
        {
            CUDAImageUtil::copy<uchar4> ( frame.m_colorIntegration, d_colorInput, m_widthIntegration, m_heightIntegration );
        }
        else
        {
            memcpy ( frame.m_colorIntegration, colorInput, sizeof ( uchar4 ) *bufferDimColorInput );

        }
    }
    else
    {
        if ( ManagedRGBDInputFrame::s_bIsOnGPU )
        {
            CUDAImageUtil::resampleUCHAR4 (frame.m_colorIntegration, m_widthIntegration, m_heightIntegration, d_colorInput, initWidth, initHeight );
        }
        else
        {
            CUDAImageUtil::resampleUCHAR4 (frame.s_colorIntegrationGlobal, m_widthIntegration, m_heightIntegration, d_colorInput, initWidth, initHeight );
            MLIB_CUDA_SAFE_CALL ( cudaMemcpy ( frame.m_colorIntegration, frame.s_colorIntegrationGlobal, sizeof ( uchar4 ) *m_widthIntegration*m_heightIntegration, cudaMemcpyDeviceToHost ) );
            frame.s_activeColorGPU = &frame;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // Process Depth
    ////////////////////////////////////////////////////////////////////////////////////
    uint depthWidth = depth.cols;
    uint depthHeight = depth.rows;
    const unsigned int bufferDimDepthInput = depthWidth * depthHeight;
    MLIB_CUDA_SAFE_CALL ( cudaMemcpy ( d_depthInputRaw, depthInput, sizeof ( float ) * depthWidth * depthHeight, cudaMemcpyHostToDevice ) );

    if ( ManagedRGBDInputFrame::s_bIsOnGPU )
    {
        CUDAImageUtil::copy<float> ( frame.m_depthRaw, d_depthInputRaw, depthWidth, depthHeight );
        //std::swap(frame.m_depthIntegration, d_depthInput);
    }
    else
    {
        MLIB_CUDA_SAFE_CALL ( cudaMemcpy ( frame.m_depthRaw, d_depthInputRaw, sizeof ( float ) *bufferDimDepthInput, cudaMemcpyDeviceToHost ) );
    }

    if ( GlobalAppState::get().s_depthFilter)
    {
        unsigned int numIter = 2;
        numIter = 2 * ( ( numIter + 1 ) / 2 );
        for ( unsigned int i = 0; i < numIter; i++ )
        {
            if ( i % 2 == 0 )
            {
                CUDAImageUtil::erodeDepthMap ( d_depthInputFiltered, d_depthInputRaw, 3,
                                               depthWidth, depthHeight, 0.05f, 0.3f );
            }
            else
            {
                CUDAImageUtil::erodeDepthMap ( d_depthInputRaw, d_depthInputFiltered, 3,
                                               depthWidth, depthHeight, 0.05f, 0.3f );
            }
        }

        CUDAImageUtil::gaussFilterDepthMap ( d_depthInputFiltered, d_depthInputRaw, GlobalAppState::get().s_depthSigmaD, GlobalAppState::get().s_depthSigmaR,
                                             depthWidth, depthHeight );
    }
    else
    {
        CUDAImageUtil::copy<float> ( d_depthInputFiltered, d_depthInputRaw, depthWidth, depthHeight );
    }

    if ( ( depthWidth == m_widthIntegration ) && ( depthHeight == m_heightIntegration ) )
    {
        if ( ManagedRGBDInputFrame::s_bIsOnGPU )
        {
            CUDAImageUtil::copy<float> ( frame.m_depthIntegration, d_depthInputFiltered, m_widthIntegration, m_heightIntegration );
            //std::swap(frame.m_depthIntegration, d_depthInput);
        }
        else
        {
            MLIB_CUDA_SAFE_CALL ( cudaMemcpy ( frame.m_depthIntegration, d_depthInputFiltered, sizeof ( float ) *bufferDimDepthInput, cudaMemcpyDeviceToHost ) );
        }
    }
    else
    {
        if ( ManagedRGBDInputFrame::s_bIsOnGPU )
        {
            CUDAImageUtil::resampleFloat ( frame.m_depthIntegration, m_widthIntegration, m_heightIntegration, d_depthInputFiltered, depthWidth, depthHeight );
        }
        else
        {
            CUDAImageUtil::resampleFloat ( frame.s_depthIntegrationGlobal, m_widthIntegration, m_heightIntegration, d_depthInputFiltered, depthWidth, depthHeight );
            MLIB_CUDA_SAFE_CALL ( cudaMemcpy ( frame.m_depthIntegration, frame.s_depthIntegrationGlobal, sizeof ( float ) *m_widthIntegration*m_heightIntegration, cudaMemcpyDeviceToHost ) );
            frame.s_activeDepthGPU = &frame;
        }
    }

    m_currFrame++;
    return true;
}

