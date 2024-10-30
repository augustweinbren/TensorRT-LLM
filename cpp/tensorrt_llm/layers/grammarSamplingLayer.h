#pragma once

#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm::layers
{

//! \brief Layer to perform constrained decoding based on a context-free grammar (CFG).
//! This layer applies CFG constraints to the logits before sampling, ensuring that the output sequences
//! adhere to the specified grammar.
template <typename T>
class GrammarSamplingLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    GrammarSamplingLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
        std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! \brief Modifies 'outputs->logits' in-place with -INF for outputs not matching grammar
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    // //! @returns workspace needed for this layer in bytes
    // [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

// protected:
//     size_t mWorkspaceSize{0};

//     // Data structure representing the context-free grammar (CFG)
//     //TODO: change this to the 
//     CFGData mCFG;

//     // Device buffer for allowed tokens per batch
//     TensorPtr mAllowedTokensDevice;

//     using Base::mDecoderDomain;

private:
    void allocateBuffer();

    // Applies CFG constraints to the logits
    void applyCFGConstraints(TensorPtr const& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<DecodingInputs> const& inputs, BufferConstPtr const& batchSlots,
        DecoderDomain const& decoderDomain, runtime::SizeType32 maxSeqLen);

    executor::DecodingMode mDecodingMode;
};

} // namespace tensorrt_llm::layers
