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
struct CFGConstraintsKernelParams
{
    //! Input buffer [batchSize, vocabSizePadded].
    //! Logits for each token in the vocabulary for each batch element.
    T* logits{nullptr};

    //! Output IDs pointers per batch [batchSize], pointing to arrays of token IDs.
    int32_t** outputIdsPtrs{nullptr};

    //! Sequence lengths per batch [batchSize].
    int32_t const* sequenceLengths{nullptr};

    //! Batch slots [batchSize], mapping batch indices to actual batch slots if needed.
    int32_t const* batchSlots{nullptr};

    //! Batch size.
    int32_t batchSize{-1};

    //! Padded vocabulary size.
    int32_t vocabSizePadded{-1};

    //! Current decoding step.
    int32_t step{-1};

    //! Device buffer [batchSize, vocabSizePadded] indicating allowed tokens per batch.
    //! Each element is a boolean indicating whether a token is allowed.
    bool* allowedTokens{nullptr};
};

// Kernel invocation function to apply CFG constraints to logits
template <typename T>
void invokeApplyCFGConstraints(CFGConstraintsKernelParams<T>& params, cudaStream_t stream);

// Structure to hold parameters for the sampling kernel with constraints
template <typename T>
struct SamplingKernelParams
{
    //! Input buffer [batchSize, vocabSizePadded].
    //! Log probabilities of each token in the vocabulary for each batch element.
    T* logProbs{nullptr};

    //! Output IDs pointers per batch [batchSize], pointing to arrays of token IDs.
    int32_t** outputIdsPtrs{nullptr};

    //! Pointer to the workspace needed for sampling.
    void* workspace{nullptr};

    //! Sequence lengths per batch [batchSize].
    int32_t const* sequenceLengths{nullptr};

    //! End IDs per batch [batchSize].
    int32_t const* endIds{nullptr};

    //! Batch slots [batchSize], mapping batch indices to actual batch slots if needed.
    int32_t const* batchSlots{nullptr};

    //! Finished state input per batch [batchSize].
    runtime::FinishedState const* finishedInput{nullptr};

    //! Finished state output per batch [batchSize].
    runtime::FinishedState* finishedOutput{nullptr};

    //! Cumulative log probabilities per batch [batchSize].
    float* cumLogProbs{nullptr};

    //! Output log probabilities per batch [batchSize].
    float* outputLogProbs{nullptr};

    //! CURAND states per batch [batchSize].
    curandState_t* curandState{nullptr};

    //! Batch size.
    int32_t batchSize{-1};

    //! Maximum batch size.
    int32_t maxBatchSize{-1};

    //! Maximum tokens per step.
    int32_t maxTokensPerStep{-1};

    //! Padded vocabulary size.
    int32_t vocabSizePadded{-1};

    //! Flag indicating whether the logits are probabilities (true) or log probabilities (false).
    bool logitsHasProbs{false};

    //! Maximum top K value for sampling.
    int32_t maxTopK{-1};

    //! Top K values per batch [batchSize].
    int32_t const* topKs{nullptr};

    //! Maximum top P value for sampling.
    float maxTopP{1.0f};

    //! Top P values per batch [batchSize].
    float const* topPs{nullptr};

    //! Skip decode flags per batch [batchSize].
    bool const* skipDecode{nullptr};

    //! Flag to normalize log probabilities.
    bool normalizeLogProbs{false};

    //! Tokens per step per batch [batchSize], optional.
    int32_t const* tokensPerStep{nullptr};

    //! Maximum sequence length.
    int32_t maxSeqLen{-1};

    //! Flag to return all selected tokens.
    bool returnAllSelectedTokens{false};
};

// Kernel invocation function for sampling with constraints
template <typename T>
void invokeSamplingWithConstraints(SamplingKernelParams<T>& params, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
