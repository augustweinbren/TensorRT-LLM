/*
 * samplingGrammarKernels.h
 *
 * Declares the CUDA kernels and kernel invocation functions for constrained decoding.
 */

#pragma once

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/common.h"
#include <curand_kernel.h>

namespace tensorrt_llm::kernels
{

// Structure to hold parameters for the kernel that applies CFG constraints
template <typename T>
struct GrammarSamplingKernelParams
{
    //! Input buffer [batchSize, maxTokensPerStep, vocabSizePadded].
    //! Log probabilities of each token in the vocab. If logitsHasProbs is true,
    //! logProbs must contain **just** probabilities instead of log probabilities.
    T const* logProbs{nullptr};
    //! input buffer [batchSize][tokensPerStep, vocabSizePadded] array of pointers to logits.
    //! If nullptr, logProbs is used.
    T const* const* logProbsPtrs{nullptr};

    //! output buffer [maxBatchSize][maxSeqLen], optional. Contains pointers to rows
    //! with output tokens per request. If nullptr, outputIds must be provided.
    runtime::TokenIdType** outputIdsPtrs{nullptr};
    //! output buffer [maxBatchSize, maxSeqLen], optional. Tensor to store output tokens.
    //! Not used if outputIdsPtrs != nullptr
    runtime::TokenIdType* outputIds{nullptr};

    //! Required. Pointer to the workspace of size returned by getTopKWorkspaceSize.
    //! Has to be pre-allocated by caller.
    //! Function does not take ownership of the buffer
    void* workspace{nullptr};

    //! Sequence lengths per batch [batchSize].
    int32_t const* sequenceLengths{nullptr};

    //! input buffer [maxBatchSize], optional. EOS token ids per request
    runtime::TokenIdType const* endIds{nullptr};

    //! Sequence lengths per batch [batchSize].
    int32_t const* sequenceLengths{nullptr};

    //! Batch slots [batchSize], mapping batch indices to actual batch slots if needed.
    int32_t const* batchSlots{nullptr};

    //! input buffer [maxBatchSize], optional. Number of tokens per step for each request.
    //! It is assumed that all requests have maxTokensPerStep tokens per step if nullptr.
    runtime::SizeType32 const* tokensPerStep{nullptr};

    //! input buffer [maxBatchSize], optional. If true, request exits early.
    FinishedState const* finishedInput{nullptr};
    //! output buffer [maxBatchSize], optional.
    //! Set to true if sequence has finished (if finished || outputId == endId).
    FinishedState* finishedOutput{nullptr};
    //! input buffer [maxBatchSize]. Flags whether to skip decoding per request
    bool const* skipDecode{nullptr};

    //! input/output buffer [maxBatchSize], optional.
    //! Cumulative log probability of selected tokens. Ignored if nullptr
    float* cumLogProbs{nullptr};
    //! output buffer [maxBatchSize]. Log probs is the probability induced by the top-k sampling.
    //! If normalizeLogProbs is true, we normalize the probability 'expLogit' of the selected token
    //! by the probability 's_sum' of a set of top-k tokens, meaning the logProb is the probability
    //! of the selected token, conditioned on the event that it is selected,
    //! i.e., log_prob = log P(i | i is in top-k) = log(expLogit / s_sum).
    //! Ignored if nullptr.
    float* outputLogProbs{nullptr};

    //! input buffer [maxBatchSize], optional. Initialized curand states.
    //! If nullptr, 1 is always used.
    curandState_t* curandState{nullptr};

    //! Allowed tokens for the next iteration, grouped by rules (e.g. to store "a" and "ab" together)
    runtime::TokenIdType** allowedTokens{nullptr};

    //! max probability for constrained decoding sampling
    runtime::TokenIdType maxConstrained{nullptr};

    runtime::SizeType32 batchSize{-1};
    runtime::SizeType32 maxBatchSize{-1};
    runtime::SizeType32 vocabSizePadded{-1};
    runtime::SizeType32 maxTokensPerStep{-1};
    runtime::SizeType32 maxSeqLen{-1};

    //! Current decoding step.
    int32_t step{-1};

    //! when set to True outputLogProbs are normalized to Grammar
    bool normalizeLogProbs{false};
    //! flag to highlight that logProbs contains probabilities
    bool logitsHasProbs{false};
    //! flag to return all selected Grammar sampled results
    bool returnAllSelectedTokens{false};
};
    
    void checkParams() const
    {
        TLLM_CHECK(batchSize > 0);
        TLLM_CHECK(maxBatchSize > 0);
        TLLM_CHECK(maxBatchSize >= batchSize);
        TLLM_CHECK(vocabSizePadded > 0);
        TLLM_CHECK(maxTokensPerStep > 0);

        TLLM_CHECK(logProbs || logProbsPtrs);
        TLLM_CHECK(outputIds || outputIdsPtrs);

        if (maxTokensPerStep > 1)
        {
            TLLM_CHECK(tokensPerStep);
        }

        if (outputIds)
        {
            TLLM_CHECK(maxSeqLen > 0);
        }

        TLLM_CHECK(workspace);

        TLLM_CHECK(maxTokensPerStep != 1 || returnAllSelectedTokens || sequenceLengths);
        TLLM_CHECK(maxTokensPerStep != 1 || returnAllSelectedTokens || endIds);
        if (cumLogProbs != nullptr || outputLogProbs != nullptr)
        {
            TLLM_CHECK(maxTokensPerStep == 1 && !returnAllSelectedTokens);
        }
        TLLM_CHECK(((finishedOutput == nullptr) ^ (endIds == nullptr)) == 0);
    }
};

// clang-format off
//! \brief Given logProbs, performs Grammar sampling. Fills sampled tokens to outputIds.
//! Computes sequenceLength, finished state, cumLogProbs inplace.
//! Sampling per request can be controlled using skipDecode.
//! Function sets workspaceSize and exits early if workspace is nullptr.
//! If logits are Nan, we set output token to be the last in the vocabulary.
// clang-format on
template <typename T>
void invokeBatchGrammarSampling(GrammarSamplingKernelParams<T>& params, cudaStream_t stream);

template <typename T>
[[nodiscard]] std::vector<size_t> getGrammarWorkspaceSizes(runtime::SizeType32 batchSize,
    runtime::SizeType32 maxTokensPerStep, runtime::SizeType32 maxGrammar, runtime::SizeType32 vocabSizePadded)
{
    runtime::SizeType32 constexpr maxBlockPerBeam = 8;
    auto const tempLogProbsBufSize = sizeof(T) * batchSize * maxTokensPerStep * vocabSizePadded;         // type T
    auto const grammarTmpIdsBufSize
        = sizeof(runtime::SizeType32) * batchSize * maxTokensPerStep * maxGrammar * maxBlockPerBeam;        // type int
    auto const grammarTmpValBufSize = sizeof(T) * batchSize * maxTokensPerStep * maxGrammar * maxBlockPerBeam; // type T

    return {tempLogProbsBufSize, grammarTmpIdsBufSize, grammarTmpValBufSize};
}

//! \brief Returns workspace size in bytes needed for sampling TopK computation
//! \param batchSize batch size
//! \param maxTokensPerStep maximum number of tokens per computed per step
//! \param maxTopK maximum among all topKs K for topK sampling
//! \param vocabSizePadded size of padded vocab
template <typename T>
[[nodiscard]] size_t getGrammarWorkspaceSize(runtime::SizeType32 batchSize, runtime::SizeType32 maxTokensPerStep,
    runtime::SizeType32 maxGrammar, runtime::SizeType32 vocabSizePadded)
{
    auto const workspaceSizes = getGrammarWorkspaceSizes<T>(batchSize, maxTokensPerStep, maxTopK, vocabSizePadded);
    return tensorrt_llm::common::calcAlignedSize(workspaceSizes, 256);
}

void invokeSetupGrammarRuntimeArgs(runtime::SizeType32 batchSize, runtime::TokenIDType* grammar,
    runtime::SizeType32* grammarDevicePtr, runtime::SizeType32 runtimeGrammarSize,
    bool* skipDecodeDevicePtr, runtime::SizeType32 const* batchSlotsDevicePtr, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
