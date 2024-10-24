/*
 * samplingGrammarKernels.cu
 *
 * Implements the CUDA kernels for constrained decoding.
 */

#include "samplingGrammarKernels.h"
#include "tensorrt_llm/common/logger.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels
{

// Helper macro to check CUDA errors
#define CHECK_CUDA(call)                                                \
    do                                                                  \
    {                                                                   \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess)                                         \
        {                                                               \
            TLLM_LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return;                                                     \
        }                                                               \
    } while (0)

template <typename T>
__global__ void applyCFGConstraintsKernel(CFGConstraintsKernelParams<T> params)
{
    // Each block handles one batch element
    int32_t batchId = blockIdx.x;
    int32_t vocabSize = params.vocabSizePadded;
    int32_t threadId = threadIdx.x;

    if (batchId < params.batchSize)
    {
        T* logits = params.logits + batchId * vocabSize;
        int32_t* outputIds = params.outputIdsPtrs[batchId];
        int32_t sequenceLength = params.sequenceLengths[batchId];

        // Get the last token in the sequence
        int32_t lastToken = (sequenceLength > 0) ? outputIds[sequenceLength - 1] : -1;

        // For simplicity, assume that allowed tokens are stored in a device array per batch
        // In practice, you may need to pass additional parameters or precompute allowed tokens
        bool* allowedTokens = params.allowedTokens + batchId * vocabSize;

        // Each thread processes one token in the vocabulary
        for (int32_t tokenId = threadId; tokenId < vocabSize; tokenId += blockDim.x)
        {
            // Check if the token is allowed
            bool isAllowed = allowedTokens[tokenId];

            // If not allowed, set the logit to -FLT_MAX
            if (!isAllowed)
            {
                logits[tokenId] = -FLT_MAX;
            }
        }
    }
}

template <typename T>
void invokeApplyCFGConstraints(CFGConstraintsKernelParams<T>& params, cudaStream_t stream)
{
    int32_t threadsPerBlock = 256;
    int32_t blocksPerGrid = params.batchSize;

    applyCFGConstraintsKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(params);

    // Check for errors
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void invokeSamplingWithConstraints(SamplingKernelParams<T>& params, cudaStream_t stream)
{
    // Reuse the existing sampling kernel from samplingTopKKernels.cu
    // Adjust it to operate on the constrained logits

    // Set up parameters similar to TopKSamplingKernelParams
    TopKSamplingKernelParams<T> topKParams;
    topKParams.logProbs = params.logProbs;
    topKParams.outputIdsPtr = params.outputIdsPtrs;
    topKParams.workspace = params.workspace;
    topKParams.sequenceLengths = params.sequenceLengths;
    topKParams.endIds = params.endIds;
    topKParams.batchSlots = params.batchSlots;
    topKParams.finishedInput = params.finishedInput;
    topKParams.finishedOutput = params.finishedOutput;
    topKParams.cumLogProbs = params.cumLogProbs;
    topKParams.outputLogProbs = params.outputLogProbs;
    topKParams.curandState = params.curandState;
    topKParams.batchSize = params.batchSize;
    topKParams.maxBatchSize = params.maxBatchSize;
    topKParams.maxTokensPerStep = params.maxTokensPerStep;
    topKParams.vocabSizePadded = params.vocabSizePadded;
    topKParams.logitsHasProbs = params.logitsHasProbs;
    topKParams.maxTopK = params.maxTopK;
    topKParams.topKs = params.topKs;
    topKParams.maxTopP = params.maxTopP;
    topKParams.topPs = params.topPs;
    topKParams.skipDecode = params.skipDecode;
    topKParams.normalizeLogProbs = params.normalizeLogProbs;
    topKParams.tokensPerStep = params.tokensPerStep;
    topKParams.maxSeqLen = params.maxSeqLen;
    topKParams.returnAllSelectedTokens = params.returnAllSelectedTokens;

    // Now invoke the existing sampling kernel
    invokeBatchTopKSampling(topKParams, stream);
}

// Explicit template instantiation
template void invokeApplyCFGConstraints<float>(CFGConstraintsKernelParams<float>& params, cudaStream_t stream);
template void invokeApplyCFGConstraints<half>(CFGConstraintsKernelParams<half>& params, cudaStream_t stream);

template void invokeSamplingWithConstraints<float>(SamplingKernelParams<float>& params, cudaStream_t stream);
template void invokeSamplingWithConstraints<half>(SamplingKernelParams<half>& params, cudaStream_t stream);

} // namespace tensorrt_llm::kernels

#undef CHECK_CUDA
