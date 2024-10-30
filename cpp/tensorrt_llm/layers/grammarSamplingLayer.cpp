#include "grammarSamplingLayer.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingGrammarKernels.h"
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
    executor::DecodingMode const& mode,
    DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager),
    mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GrammarSamplingLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mDecodingMode.isUseContextFreeGrammar()) {
        mAllowedTokensDevice 
            = mBufferManager->gpu(ITensor::makeShape({mDecoderDomain.getBatchSize()}), TRTDataType<SizeType32>::value);
    }


    // // Allocate any additional buffers needed for constrained decoding
    // mWorkspaceSize = getGrammarSamplingWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded());

    // For example, allocate buffer to store allowed tokens per batch
    mAllowedTokens = mBufferManager->pinnedPool(
        ITensor::makeShape({mDecoderDomain.getBatchSize()}), TRTDataType<SizeType32>::value);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void GrammarSamplingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // Setup any CFG-related parameters here
    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);
    auto const& grammarParams = setupParams->grammarParams
    TLLM_CHECK_WITH_INFO(grammarParams, "grammarParams not set for setup")
    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

// Applies CFG constraints to the logits
void applyCFGConstraints(TensorPtr const& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, BufferConstPtr const& batchSlots,
    DecoderDomain const& decoderDomain, runtime::SizeType32 maxSeqLen) 
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxCFGLength = inputs->constrainedDecodingInputs->maxGrammarLen;
    if (maxCFGLength != 0)
    {
        // Temporary variables to store dereferenced inputs
        auto grammarPtr = bufferCast<TokenIdType const*>(*inputs->constrainedDecodingInputs->grammarPtr.value());
        auto grammarLens = bufferCast<SizeType32>(*inputs->constrainedDecodingInputs->grammarLengths.value());
        auto logitsPtr = bufferCast<T>(*logits);
        auto outputIdsPtr = bufferCast<TokenIdType const*>(*outputs->outputIdsPtr);
        auto parentIdsPtr
            = decoderDomain.getBeamWidth() > 1 ? bufferCast<SizeType32 const*>(*outputs->parentIdsPtr) : nullptr;
        auto sequenceLengthPtr = bufferCast<SizeType32>(*outputs->sequenceLength.value());
        auto batchSlotsPtr = bufferCast<SizeType32>(*batchSlots);

        // Call to invokeConstrainedDecoding with dereferenced inputs
        invokeApplyCFGConstraints(logitsPtr, outputIdsPtr, parentIdsPtr, batchSlotsPtr, decoderDomain.getBatchSize(),
            decoderDomain.getBeamWidth(), grammarPtr, grammarLens, maxGrammarLen,
            decoderDomain.getVocabSizePadded(), sequenceLengthPtr, maxSeqLen, getStream());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);

}

template <typename T>
void GrammarSamplingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<BaseDecodingOutputs>(baseOutputs);
    
    TLLM_CHECK_WITH_INFO(inputs->constrainedDecodingInputs, "constrainedDecodingInputs for forward is not set");

    auto const localDecoderDomain = getLocalDecoderDomain(inputs, mDecoderDomain);
    auto const maxSeqLen = outputs->outputIds->getDimension<-1>();

    applyCFGConstraints(workspace->getDeviceRuntimeLogits(), outputs, inputs, workspace->getDeviceBatchSlots(),
        getLocalDecoderDomain, maxSeqLen);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
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
