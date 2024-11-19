/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "llama-grammar.h"

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

int llama_grammar(const std::string &grammar_str) {
    std::string grammar_str_temp = 
        R"""(
            root ::= expr
            expr ::= term ("+" term)*
            term ::= number
            number ::= [0-9]+)"""
    auto * grammar = llama_grammar_init_impl(nullptr, grammar_str_temp.c_str(), "root");
    //
    

    // Save the original grammar stacks so that we can reset after every new string we want to test
    const llama_grammar_stacks stacks_org = llama_grammar_get_stacks(grammar);

    llama_grammar_stacks & stacks_cur = llama_grammar_get_stacks(grammar);

    const llama_grammar_rules & rules = llama_grammar_get_rules(grammar);
    const llama_grammar_stacks & stacks_cur = llama_grammar_get_stacks(grammar);

    const auto cpts = unicode_cpts_from_utf8("1+2+3+4+5");

    for (const auto & cpt : cpts) {
        const llama_grammar_stacks stacks_prev = llama_grammar_get_stacks(grammar); // copy

        llama_grammar_accept(rules, stacks_prev, cpt, stacks_cur);

        if (stacks_cur.empty()) {
            // no stacks means that the grammar failed to match at this point
            std::out << "false";
        }
    }

    for (const auto & stack : stacks_cur) {
        if (stack.empty()) {
            std::out << "true";
        }
    }
    // √
    llama_grammar_free_impl(grammar);
    return;

}

llama_token_data_array logitsToLlamaTokenDataArray(float* logitsCpu, size_t n_vocab) {
    // AW NOTE: from `llama_sampler_sample`
    // TODO: do not allocate each time
    std::vector<llama_token_data> cur;
    cur.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur.emplace_back(llama_token_data{token_id, logitsCpu[token_id], 0.0f});
    }

    llama_token_data_array cur_p = {
        /* .data       = */ cur.data(),
        /* .size       = */ cur.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };
    return cur_p;
}

/// @brief 
/// @param dataArray 
/// @return 
/// NOTE: this might not be necessary
std::vector<float> llamaTokenDataArrayToLogits(llama_token_data_array& dataArray) {
    std::vector<float> output;
    output.reserve(dataArray.size);
    for (auto& element : dataArray) {
        output.insert(element);
        //TODO: finish this portion
    }
}

///TODO: implement apply and accept on every callback



int main(int argc, char* argv[])
{
    // Register the TRT-LLM plugins
    initTrtLlmPlugins();

    if (argc != 2)
    {
        TLLM_LOG_ERROR("Usage: %s <dir_with_engine_files>", argv[0]);
        return 1;
    }

    std::vector<int> const sentinels = {
        2201, 128009, 9891, 128001, 32559, 
        220, 
        914, 
        339
    };
    int step = 0;

    auto logitsPostProcessorFnOld
        = [&step, &sentinels](tle::IdType reqId, tle::Tensor& logits, tle::BeamTokens const& tokens,
              tle::StreamPtr const& streamPtr, std::optional<tle::IdType> clientId)
    {
        auto logitsDataType = logits.getDataType();
        auto logitsCpu = tensorrt_llm::executor::Tensor::cpu(logitsDataType, logits.getShape());
        logitsCpu.setFrom(logits, streamPtr);
        auto* dataPtr = logitsCpu.getData();
        auto* dataPtrFloat = static_cast<float*>(dataPtr);
        for (size_t i = 0; i < logitsCpu.getSize(); ++i)
        {
            bool ban = true;
            for (int j = 0; j < sentinels.size(); j++) {
                if (i == sentinels[j]) {
                    ban = false;
                    break;
                }
            }
            if (ban) {
                dataPtrFloat[i] = -1.0e20;
            } else {
                // dataPtrFloat[i] = 0.0f;
            }
            // dataPtrFloat[i] = ban ? -1.0e20 : 0.0f;
        }
        // dataPtrFloat[sentinels[step]] = 0.0f;

        logits.setFrom(logitsCpu, streamPtr);
        step++;
        // if (step > 2) {
        //     break;
        // }
        // step = step % 3;
    };
#include <string>
#include <vector>
#include <cassert>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
    auto logitsPostProcessorFn
        = [&sentinels](tle::IdType reqId, tle::Tensor& logits, tle::BeamTokens const& tokens,
              tle::StreamPtr const& streamPtr, std::optional<tle::IdType> clientId)
    {
        auto logitsDataType = logits.getDataType();
        auto logitsCpu = tensorrt_llm::executor::Tensor::cpu(logitsDataType, logits.getShape());
        logitsCpu.setFrom(logits, streamPtr);
        auto* dataPtr = logitsCpu.getData();
        auto* dataPtrFloat = static_cast<float*>(dataPtr);
        for (size_t i = 0; i < logitsCpu.getSize(); ++i)
        {
            bool ban = true;
            for (int j = 0; j < sentinels.size(); j++) {
                if (i == sentinels[j]) {
                    ban = false;
                    break;
                }
            }
            if (ban) {
                dataPtrFloat[i] = -1.0e20;
            } else {
                // dataPtrFloat[i] = 0.0f;
            }
            // dataPtrFloat[i] = ban ? -1.0e20 : 0.0f;
        }
        // dataPtrFloat[sentinels[step]] = 0.0f;

        logits.setFrom(logitsCpu, streamPtr);
        // if (step > 2) {
        //     break;
        // }
        // step = step % 3;
    };

    std::string logitsPostProcessorName = "MyLogitsPP";

    // Create the executor for this engine
    tle::SizeType32 beamWidth = 1;
    auto executorConfig = tle::ExecutorConfig(beamWidth);

    auto logitsProcConfig = tle::LogitsPostProcessorConfig();
    logitsProcConfig.setProcessorMap(std::unordered_map<std::string, tensorrt_llm::executor::LogitsPostProcessor>{
        {logitsPostProcessorName, logitsPostProcessorFn}});
    executorConfig.setLogitsPostProcessorConfig(logitsProcConfig);

    auto trtEnginePath = argv[1];
    auto executor = tle::Executor(trtEnginePath, tle::ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    tle::SizeType32 maxNewTokens = 20;
    tle::VecTokens inputTokens{
        128000, 
        128006, 
        882, 
        128007, 
        271, 
        46864, 
        330, 
        32559, 
        220, 
        914, 
        339, 
        1, 
        128009, 
        128006, 
        78191, 
        128007, 
        271
    };
    // maxNewTokens = 1; 
    auto request = tle::Request(inputTokens, maxNewTokens);
    request.setLogitsPostProcessorName(logitsPostProcessorName);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    // Wait for the response
    auto responses = executor.awaitResponses(requestId);

    // Get outputTokens
    // for (int n = beamWidth-1; n < responses.size(); n++) {
        auto outputTokens = responses.at(0).getResult().outputTokenIds.at(beamWidth - 1);
        // auto outputTokens = responses.at(n).getResult().outputTokenIds.at(beamWidth-1);

        TLLM_LOG_INFO("Output tokens: %s", tlc::vec2str(outputTokens).c_str());
    // }

    return 0;
}
