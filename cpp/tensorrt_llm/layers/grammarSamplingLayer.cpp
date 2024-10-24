#include "grammarSamplingLayer.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"

#include <algorithm>
#include <cfloat>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <typename T>
GrammarSamplingLayer<T>::GrammarSamplingLayer(
    DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer(mDecoderDomain.getBatchSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GrammarSamplingLayer<T>::allocateBuffer(SizeType32 const batchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // Allocate any additional buffers needed for constrained decoding
    mWorkspaceSize = getGrammarSamplingWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded());

    // For example, allocate buffer to store allowed tokens per batch
    mAllowedTokensDevice = mBufferManager->gpu(
        ITensor::makeShape({batchSize, mDecoderDomain.getVocabSizePadded()}), TRTDataType<bool>::value);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GrammarSamplingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // Setup any CFG-related parameters here
    auto setupParams = std::dynamic_pointer_cast<GrammarSamplingSetupParams>(baseSetupParams);
    mCFG = setupParams->cfg; // Assuming cfg is passed in setupParams

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GrammarSamplingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<SamplingInputs>(baseInputs);

    auto const batchSize = inputs->logits.value()->getDimension<0>();

    auto logits = bufferCastOrNull<T>(inputs->logits);
    auto const* endIds = bufferCastOrNull<TokenIdType>(inputs->endIds);

    auto* outputIdsPtr = bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr);
    auto const* sequenceLengths = bufferCastOrNull<SizeType32>(outputs->sequenceLength);
    auto const* batchSlotsDevicePtr = workspace->getDeviceBatchSlotsPtr();

    // Apply CFG constraints to logits
    applyCFGConstraints(logits, outputIdsPtr, sequenceLengths, batchSlotsDevicePtr, batchSize, inputs->step);

    // Proceed with sampling
    SamplingKernelParams<T> params;
    params.logProbs = logits;
    params.outputIdsPtrs = outputIdsPtr;
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.sequenceLengths = sequenceLengths;
    params.endIds = endIds;
    params.batchSlots = batchSlotsDevicePtr;
    params.finishedInput = reinterpret_cast<FinishedState const*>(bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished));
    params.finishedOutput = reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished));
    params.cumLogProbs = bufferCastOrNull<float>(outputs->cumLogProbs);
    params.outputLogProbs = bufferCastOrNull<float>(outputs->outputLogProbsTiled);
    params.curandState = inputs->curandStates;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.maxTokensPerStep = 1;
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.logitsHasProbs = inputs->probsComputed;

    invokeSamplingWithConstraints(params, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GrammarSamplingLayer<T>::applyCFGConstraints(T* logits, TokenIdType** outputIdsPtr,
    SizeType32 const* sequenceLengths, SizeType32 const* batchSlotsDevicePtr, SizeType32 batchSize, int32_t step)
{
    // For each batch element, apply CFG constraints to the logits
    // This can be done efficiently using a CUDA kernel
    // For simplicity, let's assume we have a kernel that does this

    // Prepare parameters for the kernel
    CFGConstraintsKernelParams<T> params;
    params.logits = logits;
    params.outputIdsPtrs = outputIdsPtr;
    params.sequenceLengths = sequenceLengths;
    params.batchSlots = batchSlotsDevicePtr;
    params.batchSize = batchSize;
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.step = step;
    params.cfg = mCFG; // Pass the CFG to the kernel

    invokeApplyCFGConstraints(params, getStream());
}

template <typename T>
size_t GrammarSamplingLayer<T>::getWorkspaceSize() const noexcept
{
    return mWorkspaceSize;
}

// Explicit template instantiation
template class GrammarSamplingLayer<float>;
template class GrammarSamplingLayer<half>;

} // namespace tensorrt_llm::layers
