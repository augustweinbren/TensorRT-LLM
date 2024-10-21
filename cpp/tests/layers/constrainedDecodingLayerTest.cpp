/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "tests/layers/baseSamplingLayerTest.h"

namespace
{

using namespace tensorrt_llm::tests::layers::sampling;
using namespace tensorrt_llm::runtime;

template <typename T>
class ConstrainedDecodingTest : public BaseSamplingLayerTest<T>
{
public:
    void SetUp() override
    {
        this->mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        this->mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(this->mStream);

        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&mDeviceProp, device);

        this->mComputeProbs = true;
    }

    void initLayer(TestSamplingParams const& params) override
    {
        auto const decodingDomain
            = tensorrt_llm::layers::DecoderDomain(this->mMaxBatchSize, 1, this->mVocabSize, this->mVocabSizePadded);
        this->mSamplingLayer = std::make_shared<tensorrt_llm::layers::SamplingLayer<T>>(
            decodingDomain, this->mBufferManager, &mDeviceProp);
    }

    void batchCopy(int32_t step) override
    {
        // First, copy logits as usual
        BaseSamplingLayerTest<T>::batchCopy(step);

        // Now, apply CFG constraints to the logits
        auto* const outputIdsPtr = bufferCast<int32_t>(*this->mOutputIdsDevice);
        auto* const batchSlotsPtr = bufferCast<int32_t>(*this->mBatchSlots);
        auto logitsDevicePtr = this->mDecodingWorkspace->getDeviceRuntimeLogits();

        for (int32_t bi = 0; bi < this->mBatchSize; ++bi)
        {
            int32_t batchSlot = batchSlotsPtr[bi];
            // Get the partial sequence for this batch
            std::vector<int32_t> partialSequence(outputIdsPtr + batchSlot * this->mMaxSeqLen,
                                                 outputIdsPtr + batchSlot * this->mMaxSeqLen + step);

            // Get the allowed tokens from the CFG
            std::set<int32_t> allowedTokens = getAllowedTokens(partialSequence);

            // Apply the constraints to the logits
            // Get the logits for this batch
            auto logitsDeviceView = ITensor::slice(logitsDevicePtr, bi, 1);

            // Copy logits to host
            auto logitsHost = this->mBufferManager->copyFrom(*logitsDeviceView, tensorrt_llm::runtime::MemoryType::kCPU);
            T* logitsHostPtr = bufferCast<T>(*logitsHost);

            // Apply the constraints
            applyCFGConstraints(logitsHostPtr, allowedTokens);

            // Copy the modified logits back to device
            this->mBufferManager->copy(*logitsHost, *logitsDeviceView);
        }
    }

private:
    std::set<int32_t> getAllowedTokens(const std::vector<int32_t>& sequence)
    {
        // Implement the CFG logic here
        // Example CFG: after token '0', only '1' is allowed; after '1', only '0' is allowed.
        std::set<int32_t> allowedTokens;
        if (!sequence.empty())
        {
            int32_t lastToken = sequence.back();
            if (lastToken == 0)
            {
                allowedTokens.insert(1);
            }
            else if (lastToken == 1)
            {
                allowedTokens.insert(0);
            }
            else
            {
                // If the last token is neither 0 nor 1, allow all tokens.
                for (int32_t i = 0; i < this->mVocabSize; ++i)
                {
                    allowedTokens.insert(i);
                }
            }
        }
        else
        {
            // At the beginning, allow tokens '0' and '1'.
            allowedTokens.insert(0);
            allowedTokens.insert(1);
        }
        return allowedTokens;
    }

    void applyCFGConstraints(T* logits, const std::set<int32_t>& allowedTokens)
    {
        // Set the logits of tokens not in allowedTokens to -FLT_MAX to exclude them.
        for (int32_t i = 0; i < this->mVocabSize; ++i)
        {
            if (allowedTokens.find(i) == allowedTokens.end())
            {
                logits[i] = -FLT_MAX;
            }
        }
    }

    struct cudaDeviceProp mDeviceProp;
};

TYPED_TEST_SUITE(ConstrainedDecodingTest, FloatAndHalfTypes);

TYPED_TEST(ConstrainedDecodingTest, CFGSimpleTest)
{
    TestSamplingParams params;
    // No need to set topK or topP since we are testing constrained decoding.

    // Expected output IDs based on CFG constraints.
    // The sequence alternates between tokens '0' and '1' due to the CFG.

    std::vector<std::set<int32_t>> expectedOutputIds{
        // step 0
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1},
        // step 1
        {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0},
        // step 2
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1},
        // step 3
        {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}
    };

    this->runTest(expectedOutputIds, params);
}

} // namespace
