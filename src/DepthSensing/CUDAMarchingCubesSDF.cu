
#include <cutil_inline.h>
#include <cutil_math.h>


#include "MarchingCubesSDFUtil.h"


__global__ void resetMarchingCubesKernel(MarchingCubesData data) 
{
	*data.d_numTriangles = 0;
}

__global__ void extractIsoSurfaceKernel(HashDataStruct hashData, RayCastData rayCastData, MarchingCubesData data) 
{
	uint idx = blockIdx.x;

	const HashEntry& entry = hashData.d_hash[idx];
	if (entry.ptr != FREE_ENTRY) {
		int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
		int3 pi = pi_base + make_int3(threadIdx);
		float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

		data.extractIsoSurfaceAtPosition(worldPos, hashData, rayCastData);
	}
}
__global__ void extractObjectSurfaceKernel(HashDataStruct hashData, RayCastData rayCastData, MarchingCubesData data, uchar objectCat)
{
	uint idx = blockIdx.x;

	const HashEntry& entry = hashData.d_hash[idx];

	if (entry.ptr != FREE_ENTRY) {
		int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
		int3 pi = pi_base + make_int3(threadIdx);
		float3 worldPos = hashData.virtualVoxelPosToWorld(pi);
		Voxel v = hashData.getVoxel(worldPos);
		if(v.catLabel == objectCat)
		{
			data.extractIsoSurfaceAtPosition(worldPos, hashData, rayCastData);
		}
	}
}
__global__ void clearObjectSurfaceKernel(HashDataStruct hashData, RayCastData rayCastData, MarchingCubesData data, uchar objectCat)
{
    uint idx = blockIdx.x;

    const HashEntry& entry = hashData.d_hash[idx];

    if (entry.ptr != FREE_ENTRY) {
        int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
        int3 pi = pi_base + make_int3(threadIdx);
        float3 worldPos = hashData.virtualVoxelPosToWorld(pi);
        Voxel v = hashData.getVoxel(worldPos);
        if(v.catLabel == objectCat)
        {
            int id = entry.ptr + hashData.virtualVoxelPosToLocalSDFBlockIndex(pi);
            hashData.deleteVoxel(id);
        }
    }
}
__global__ void extractDynamicSurfaceKernel(HashDataStruct hashData, RayCastData rayCastData, MarchingCubesData data)
{
    uint idx = blockIdx.x;

    const HashEntry& entry = hashData.d_hash[idx];
    if (entry.ptr != FREE_ENTRY) {
        int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
        int3 pi = pi_base + make_int3(threadIdx);
        float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

        data.extractDynamicSurfaceAtPosition(worldPos, hashData, rayCastData);
    }
}


extern "C" void resetMarchingCubesCUDA(MarchingCubesData& data)
{
	const dim3 blockSize(1, 1, 1);
	const dim3 gridSize(1, 1, 1);

	resetMarchingCubesKernel<<<gridSize, blockSize>>>(data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

extern "C" void extractIsoSurfaceCUDA(const HashDataStruct& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data)
{
	const dim3 gridSize(params.m_hashNumBuckets*params.m_hashBucketSize, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

	extractIsoSurfaceKernel<<<gridSize, blockSize>>>(hashData, rayCastData, data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}
extern "C" void extractObjectSurfaceCUDA(const HashDataStruct& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data, uchar objCat)
{
	const dim3 gridSize(params.m_hashNumBuckets*params.m_hashBucketSize, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

	extractObjectSurfaceKernel<<<gridSize, blockSize>>>(hashData, rayCastData, data, objCat);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}
extern "C" void clearObjectSurfaceCUDA(const HashDataStruct& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data, uchar objCat)
{
    const dim3 gridSize(params.m_hashNumBuckets*params.m_hashBucketSize, 1, 1);
    const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

    clearObjectSurfaceKernel<<<gridSize, blockSize>>>(hashData, rayCastData, data, objCat);

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}
extern "C" void extractDynamicSurfaceCUDA(const HashDataStruct& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data)
{
    const dim3 gridSize(params.m_hashNumBuckets*params.m_hashBucketSize, 1, 1);
    const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

    extractDynamicSurfaceKernel<<<gridSize, blockSize>>>(hashData, rayCastData, data);

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}