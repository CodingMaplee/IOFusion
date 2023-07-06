#pragma once

#include "CUDAImageUtil.h"
#include "CUDAImageCalibrator.h"
#include "TimingLog.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

class CUDAImageManager {
public:

	class ManagedRGBDInputFrame {
	public:
		friend class CUDAImageManager;

		static void globalInit(unsigned int width, unsigned int height, unsigned int widthRaw, unsigned int heightRaw, bool isOnGPU)
		{
			globalFree();

			s_width = width;
            s_widthRaw = widthRaw;
			s_height = height;
            s_heightRaw = heightRaw;
			s_bIsOnGPU = isOnGPU;

			if (!s_bIsOnGPU) {
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&s_depthIntegrationGlobal, sizeof(float)*width*height));
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&s_colorIntegrationGlobal, sizeof(uchar4)*width*height));
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&s_depthRawGlobal, sizeof(float)*s_widthRaw*s_heightRaw));
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&s_colorRawGlobal, sizeof(uchar4)*s_widthRaw*s_heightRaw));
            }
			else {
				s_depthIntegrationGlobal = new float[width*height];
				s_colorIntegrationGlobal = new uchar4[width*height];
                s_depthRawGlobal = new float[s_widthRaw*s_heightRaw];
                s_colorRawGlobal = new uchar4[s_widthRaw*s_heightRaw];
			}
		}
		static void globalFree()
		{
			if (!s_bIsOnGPU) {
				MLIB_CUDA_SAFE_FREE(s_depthIntegrationGlobal);
				MLIB_CUDA_SAFE_FREE(s_colorIntegrationGlobal);
			}
			else {
				SAFE_DELETE_ARRAY(s_depthIntegrationGlobal);
				SAFE_DELETE_ARRAY(s_colorIntegrationGlobal);
			}
		}


		void alloc() {
			if (s_bIsOnGPU) {
				//printf("something\n");
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_depthIntegration, sizeof(float)*s_width*s_height));
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_depthRaw, sizeof(float)*s_widthRaw*s_heightRaw));
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_colorIntegration, sizeof(uchar4)*s_width*s_height));
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_colorRaw, sizeof(uchar4)*s_widthRaw*s_heightRaw));
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_dynamicBoxIntegration, sizeof(uint) * s_width * s_height));
                MLIB_CUDA_SAFE_CALL(cudaMemset(m_dynamicBoxIntegration, 0, sizeof(uint) * s_width * s_height));
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_dynamicBoxErodeIntegration, sizeof(uint) * s_width * s_height));
                MLIB_CUDA_SAFE_CALL(cudaMemset(m_dynamicBoxErodeIntegration, 0, sizeof(uint) * s_width * s_height));
			}
			else {
				m_depthIntegration = new float[s_width*s_height];
                m_depthRaw = new float[s_widthRaw*s_heightRaw];
				m_colorIntegration = new uchar4[s_width*s_height];
                m_colorRaw = new uchar4[s_widthRaw*s_heightRaw];

                uint* dynamicBoxIntegration;
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&dynamicBoxIntegration, sizeof(uint) * s_width * s_height));
                MLIB_CUDA_SAFE_CALL(cudaMemset(dynamicBoxIntegration, 0, sizeof(uint) * s_width * s_height));
                m_dynamicBoxIntegration = new uint[s_width * s_height];
                MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_dynamicBoxIntegration, dynamicBoxIntegration, sizeof(uint) * s_width * s_height, cudaMemcpyDeviceToHost));
                MLIB_CUDA_SAFE_CALL(cudaFree(dynamicBoxIntegration));

                uint* dynamicBoxErodeIntegration;
                MLIB_CUDA_SAFE_CALL(cudaMalloc(&dynamicBoxErodeIntegration, sizeof(uint) * s_width * s_height));
                MLIB_CUDA_SAFE_CALL(cudaMemset(dynamicBoxErodeIntegration, 0, sizeof(uint) * s_width * s_height));
                m_dynamicBoxErodeIntegration = new uint[s_width * s_height];
                MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_dynamicBoxErodeIntegration, dynamicBoxErodeIntegration, sizeof(uint) * s_width * s_height, cudaMemcpyDeviceToHost));
                MLIB_CUDA_SAFE_CALL(cudaFree(dynamicBoxErodeIntegration));
			}
		}


		void free() {
			if (s_bIsOnGPU) {
				MLIB_CUDA_SAFE_FREE(m_depthIntegration);
				MLIB_CUDA_SAFE_FREE(m_colorIntegration);
                MLIB_CUDA_SAFE_FREE(m_depthRaw);
                MLIB_CUDA_SAFE_FREE(m_colorRaw);
                MLIB_CUDA_SAFE_FREE(m_dynamicBoxIntegration);
                MLIB_CUDA_SAFE_FREE(m_dynamicBoxErodeIntegration);
			}
			else {
				SAFE_DELETE_ARRAY(m_depthIntegration);
				SAFE_DELETE_ARRAY(m_colorIntegration);
                SAFE_DELETE_ARRAY(m_depthRaw);
                SAFE_DELETE_ARRAY(m_colorRaw);
                SAFE_DELETE_ARRAY(m_dynamicBoxIntegration);
                SAFE_DELETE_ARRAY(m_dynamicBoxErodeIntegration);
			}
		}

        float* getDepthFramePtr()
        {
            return m_depthIntegration;
        }


		const float* getDepthFrameGPU() {	//be aware that only one depth frame is globally valid at a time
			if (s_bIsOnGPU) {
				return m_depthIntegration;
			}
			else {
				if (this != s_activeDepthGPU) {
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_depthIntegrationGlobal, m_depthIntegration, sizeof(float)*s_width*s_height, cudaMemcpyHostToDevice));
					s_activeDepthGPU = this;
				}
				return s_depthIntegrationGlobal;
			}
		}
		const uchar4* getColorFrameGPU() {	//be aware that only one depth frame is globally valid at a time
			if (s_bIsOnGPU) {
				return m_colorIntegration;
			}
			else {
				if (this != s_activeColorGPU) {
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_colorIntegrationGlobal, m_colorIntegration, sizeof(uchar4)*s_width*s_height, cudaMemcpyHostToDevice));
					s_activeColorGPU = this;
				}
				return s_colorIntegrationGlobal;
			}
		}
        const uchar4* getColorRawGPU() {	//be aware that only one depth frame is globally valid at a time
            if (s_bIsOnGPU) {
                return m_colorRaw;
            }
            else {
                if (this != s_activeColorGPU) {
                    MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_colorRawGlobal, m_colorRaw, sizeof(uchar4)*s_widthRaw*s_heightRaw, cudaMemcpyHostToDevice));
                    s_activeColorGPU = this;
                }
                return s_colorRawGlobal;
            }
        }

		const float* getDepthFrameCPU() {
			if (s_bIsOnGPU) {
				if (this != s_activeDepthCPU) {
					//std::cout<<"getDepthFrameCPU s_activeDepthGPU Model. \n\n"<<std::endl;
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_depthIntegrationGlobal, m_depthIntegration, sizeof(float)*s_width*s_height, cudaMemcpyDeviceToHost));
					s_activeColorCPU = this;
				}
				//std::cout<<"getDepthFrameCPU s_activeDepthCPU Model. \n\n\n"<<std::endl;
				return s_depthIntegrationGlobal;
			}
			else {
			  //std::cout<<"getDepthFrameCPU Not s_bIsOnGPU Model. \n\n\n"<<std::endl;
				return m_depthIntegration;
			}
		}
        const float* getDepthRawCPU() {
            if (s_bIsOnGPU) {
                if (this != s_activeDepthCPU) {
                    //std::cout<<"getDepthFrameCPU s_activeDepthGPU Model. \n\n"<<std::endl;
                    MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_depthRawGlobal, m_depthRaw, sizeof(float)*s_widthRaw*s_heightRaw, cudaMemcpyDeviceToHost));
                    s_activeColorCPU = this;
                }
                //std::cout<<"getDepthFrameCPU s_activeDepthCPU Model. \n\n\n"<<std::endl;
                return s_depthRawGlobal;
            }
            else {
                //std::cout<<"getDepthFrameCPU Not s_bIsOnGPU Model. \n\n\n"<<std::endl;
                return m_depthRaw;
            }
        }
		const uchar4* getColorFrameCPU() {
			if (s_bIsOnGPU) {
				if (this != s_activeColorCPU) {
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_colorIntegrationGlobal, m_colorIntegration, sizeof(uchar4)*s_width*s_height, cudaMemcpyDeviceToHost));
					s_activeDepthCPU = this;
				}
				return s_colorIntegrationGlobal;
			}
			else {
				return m_colorIntegration;
			}
		}
        const uchar4* getColorRawCPU() {
            if (s_bIsOnGPU) {
                if (this != s_activeColorCPU) {
                    MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_colorRawGlobal, m_colorRaw, sizeof(uchar4)*s_widthRaw*s_heightRaw, cudaMemcpyDeviceToHost));
                    s_activeDepthCPU = this;
                }
                return s_colorRawGlobal;
            }
            else {
                return m_colorRaw;
            }
        }

	private:
		float*	m_depthIntegration;	//either on the GPU or CPU
        float*	m_depthRaw;	//either on the GPU or CPU
		uchar4*	m_colorIntegration;	//either on the GPU or CPU
        uchar4*	m_colorRaw;	//either on the GPU or CPU
        uint*	m_dynamicBoxIntegration;	//either on the GPU or CPU
        uint*	m_dynamicBoxErodeIntegration;	//either on the GPU or CPU

		static bool			s_bIsOnGPU;

		static unsigned int s_width;
		static unsigned int s_height;
        static unsigned int s_widthRaw;
        static unsigned int s_heightRaw;

        static float*		s_depthRawGlobal;
        static uchar4*		s_colorRawGlobal;
		static float*		s_depthIntegrationGlobal;
		static uchar4*		s_colorIntegrationGlobal;
		static ManagedRGBDInputFrame*	s_activeColorGPU;
		static ManagedRGBDInputFrame*	s_activeDepthGPU;

		static float*		s_depthIntegrationGlobalCPU;
		static uchar4*		s_colorIntegrationGlobalCPU;
		static ManagedRGBDInputFrame*	s_activeColorCPU;
		static ManagedRGBDInputFrame*	s_activeDepthCPU;
	};

	CUDAImageManager(unsigned int widthIntegration, unsigned int heightIntegration,
                     unsigned int width, unsigned int height, Eigen::Matrix4f intrinsics,
                     float imageScale, bool storeFramesOnGPU = false) {

		m_widthIntegration = widthIntegration;
		m_heightIntegration = heightIntegration;
        m_widthRaw = width;
        m_heightRaw = height;
		const unsigned int bufferDimDepthInput = width*height;
		const unsigned int bufferDimColorInput = width*height;

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthInputRaw, sizeof(float)*bufferDimDepthInput));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthInputFiltered, sizeof(float)*bufferDimDepthInput));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colorInput, sizeof(uchar4)*bufferDimColorInput));
		m_currFrame = 0;
		const unsigned int rgbdSensorWidthDepth = width;
		const unsigned int rgbdSensorHeightDepth = height;

		// adapt intrinsics
		m_depthIntrinsics = intrinsics;
		m_depthIntrinsics(0,0) *= (float)m_widthIntegration / (float)rgbdSensorWidthDepth;  //focal
		m_depthIntrinsics(1,1) *= (float)m_heightIntegration/ (float)rgbdSensorHeightDepth;
		m_depthIntrinsics(0,2) *= (float)(m_widthIntegration-1) / (float)(rgbdSensorWidthDepth-1);	//principal point
		m_depthIntrinsics(1,2) *= (float)(m_heightIntegration-1) / (float)(rgbdSensorHeightDepth-1);
//        std::cout<<m_depthIntrinsics(0,0)<<" "<<m_depthIntrinsics(0,1)<<" "<<m_depthIntrinsics(0,2)<<" "<<m_depthIntrinsics(0,3)<<std::endl;
//        std::cout<<m_depthIntrinsics(1,0)<<" "<<m_depthIntrinsics(1,1)<<" "<<m_depthIntrinsics(1,2)<<" "<<m_depthIntrinsics(1,3)<<std::endl;
//        std::cout<<m_depthIntrinsics(2,0)<<" "<<m_depthIntrinsics(2,1)<<" "<<m_depthIntrinsics(2,2)<<" "<<m_depthIntrinsics(2,3)<<std::endl;
//        std::cout<<m_depthIntrinsics(3,0)<<" "<<m_depthIntrinsics(3,1)<<" "<<m_depthIntrinsics(3,2)<<" "<<m_depthIntrinsics(3,3)<<std::endl;
		m_depthIntrinsicsInv = m_depthIntrinsics.inverse();

		const unsigned int rgbdSensorWidthColor = width;
		const unsigned int rgbdSensorHeightColor = height;

		m_colorIntrinsics = intrinsics;
		m_colorIntrinsics(0,0) *= (float)m_widthIntegration / (float)rgbdSensorWidthColor;  //focal
		m_colorIntrinsics(1,1) *= (float)m_heightIntegration/ (float)rgbdSensorHeightColor;
		m_colorIntrinsics(0,2) *=  (float)(m_widthIntegration-1) / (float)(rgbdSensorWidthColor-1);	//principal point
		m_colorIntrinsics(1,2) *= (float)(m_heightIntegration-1) / (float)(rgbdSensorHeightColor-1);
		m_colorIntrinsicsInv = m_colorIntrinsics.inverse();

		// adapt extrinsics
        d_imageScale = imageScale;
		ManagedRGBDInputFrame::globalInit(getIntegrationWidth(), getIntegrationHeight(), width, height, storeFramesOnGPU);

	}
    bool readDepthAndColor ( const cv::Mat& rgbTest, const cv::Mat& depthTest, float* depthFloat,
                             vec4uc* colorRGBX, double depthFactor )
    {
        bool hr = true;

        if ( rgbTest.empty() )
        {
            std::cout << "no rgb!" << std::endl;
            hr = false;
        }
        if ( depthTest.empty() )
        {
            std::cout << "no depth!" << std::endl;
            hr = false;
        }

        if ( rgbTest.empty() || depthTest.empty() )
        {
            return false;
        }

        const float* pDepth = ( const float* ) depthTest.data;
        const uint8_t* pImage = ( const uint8_t* ) rgbTest.data;

        if ( !depthTest.empty() && !rgbTest.empty() )
        {
            unsigned int width = depthTest.cols;
            unsigned int nPixels = depthTest.cols * depthTest.rows;

            for ( unsigned int i = 0; i < nPixels; i++ )
            {
                const int x = i % width;
                const int y = i / width;
                const int src = y * width + ( x );//width - 1 -
                const float& p = pDepth[src];
                float dF = ( float ) p / (float)depthFactor;
                if ( dF >= GlobalAppState::get().s_sensorDepthMin && dF <= GlobalAppState::get().s_sensorDepthMax )
                {
                    depthFloat[i] = dF;
                }
                else
                {
                    depthFloat[i] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        // check if we need to draw depth frame to texture
        //if (m_depthFrame.isValid() && m_colorFrame.isValid())
        if ( !depthTest.empty() && !rgbTest.empty() )
        {
            //unsigned int width = m_colorFrame.getWidth();
            //unsigned int height = m_colorFrame.getHeight();
            unsigned int width = rgbTest.cols;
            unsigned int height = rgbTest.rows;
            unsigned int nPixels = width*height;

            for ( unsigned int i = 0; i < nPixels; i++ )
            {
                const int x = i%width;
                const int y = i / width;

                int y2 = y;

                if ( y2 >= 0 && y2 < ( int ) height )
                {
                    unsigned int Index1D = y2*width + ( x );	//x-flip here  width - 1 -

                    //const openni::RGB888Pixel& pixel = pImage[Index1D];

                    unsigned int c = 0;
                    c |= pImage[3*Index1D + 0];
                    c <<= 8;
                    c |= pImage[3*Index1D + 1];
                    c <<= 8;
                    c |= pImage[3*Index1D + 2];
                    c |= 0xFF000000;

                    ( ( LONG* ) colorRGBX ) [y*width + x] = c;
                }
            }
        }


        return hr;
    }

	~CUDAImageManager() {
		reset();

		MLIB_CUDA_SAFE_FREE(d_depthInputRaw);
		MLIB_CUDA_SAFE_FREE(d_depthInputFiltered);
		MLIB_CUDA_SAFE_FREE(d_colorInput);
		//m_imageCalibrator.OnD3D11DestroyDevice();

		ManagedRGBDInputFrame::globalFree();
	}

	void reset() {
		for (auto& f : m_data) {
			f.free();
		}
		m_data.clear();
	}
	bool process(cv::Mat& rgb, cv::Mat& depth, double depthFactor);

	ManagedRGBDInputFrame& getLastIntegrateFrame() {
		return m_data.back();
	}

	ManagedRGBDInputFrame& getIntegrateFrame(unsigned int frame) {
		return m_data[frame];
	}

	// called after process
	unsigned int getCurrFrameNumber() const {
		MLIB_ASSERT(m_currFrame > 0);
		return m_currFrame - 1;
	}

	unsigned int getIntegrationWidth() const {
		return m_widthIntegration;
	}
	unsigned int getIntegrationHeight() const {
		return m_heightIntegration;
	}
    unsigned int getRawWidth() const {
        return m_widthRaw;
    }
    unsigned int getRawHeight() const {
        return m_heightRaw;
    }
	const Eigen::Matrix4f & getDepthIntrinsics() const	{
		return m_depthIntrinsics;
	}

	const Eigen::Matrix4f& getDepthIntrinsicsInv() const {
		return m_depthIntrinsicsInv;
	}

	//const mat4f& getColorIntrinsics() const	{
	//	return m_colorIntrinsics;
	//}

	//const mat4f& getColorIntrinsicsInv() const {
	//	return m_colorIntrinsicsInv;
	//}

private:
	CUDAImageCalibrator m_imageCalibrator;

    Eigen::Matrix4f m_colorIntrinsics;
    Eigen::Matrix4f m_colorIntrinsicsInv;
    Eigen::Matrix4f m_depthIntrinsics;
    Eigen::Matrix4f m_depthIntrinsicsInv;

	//! resolution for integration both depth and color data
	unsigned int m_widthIntegration;
	unsigned int m_heightIntegration;
    unsigned int m_widthRaw;
    unsigned int m_heightRaw;
	//! temporary GPU storage for inputting the current frame
	float*	d_depthInputRaw;
	uchar4*	d_colorInput;
	float*	d_depthInputFiltered;
    float d_imageScale;
	//! all image data on the GPU
	std::vector<ManagedRGBDInputFrame> m_data;

	unsigned int m_currFrame;

};