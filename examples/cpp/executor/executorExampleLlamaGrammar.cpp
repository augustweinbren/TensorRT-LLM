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
#include <string>
#include <vector>
#include <cassert>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

/*
PSEUDOCODE:
1. Initialize and return Llama grammar using string
3. OutputToken = -1
4. While OutputToken != End of Grammar (llama_token_is_eog):
    - convert tokens to llama grammar format
    - llama_grammar_apply on converted tokens (changes the logits)
    - update original logits
    - llama_grammar_accept
5. grammar_free

*/


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

/// @brief Modifies the `logitsTRTFormat` float array in-place based on the `constrainingGrammar`
/// @param logitsTRTFormat
/// @param logitsCount 
/// @param constrainingGrammar 
void constrainLogitsToGrammar(float* logitsTRTFormat, size_t logitsCount,  llama_grammar* constrainingGrammar) {
    llama_token_data_array logitsLlamaFormat = logitsToLlamaTokenDataArray(logitsTRTFormat, logitsCount);

    llama_grammar_apply_impl(constrainingGrammar, &logitsLlamaFormat);
    for (int i = 0; i < logitsLlamaFormat.size; i++){
        //get token id
        llama_token tokenId = logitsLlamaFormat.data[i].id;
        // update logitsTRTFormat at tokenId slot with logitsLlamaFormat's logit
        logitsTRTFormat[tokenId] = logitsLlamaFormat.data[i].logit;
    }
    return;
}
    // logitsLlamaFormat.size
    // for (const auto & cpt : cpts) {
    //     const llama_grammar_stacks stacks_prev = llama_grammar_get_stacks(grammar); // copy

    //     llama_grammar_accept(rules, stacks_prev, cpt, stacks_cur);

    //     if (stacks_cur.empty()) {
    //         // no stacks means that the grammar failed to match at this point
    //         std::out << "false";
    //     }
    // }

    // for (const auto & stack : stacks_cur) {
    //     if (stack.empty()) {
    //         std::out << "true";
    //     }
    // }
    // // √
    // llama_grammar_free_impl(grammar);

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
    

    static const char* sumGrammar = 
        R"""(
            root ::= expr
            expr ::= term ("+" term)*
            term ::= number
            number ::= [0-9]+)""";
    static llama_grammar * constrainingGrammar = llama_grammar_init_impl(nullptr, sumGrammar, "root");
    int step = 0;
    auto logitsPostProcessorFn
        = [&constrainingGrammar](tle::IdType reqId, tle::Tensor& logits, tle::BeamTokens const& tokens,
              tle::StreamPtr const& streamPtr, std::optional<tle::IdType> clientId)
    {
        auto logitsDataType = logits.getDataType();
        auto logitsCpu = tensorrt_llm::executor::Tensor::cpu(logitsDataType, logits.getShape());
        logitsCpu.setFrom(logits, streamPtr);
        auto* logitsUnformatted = logitsCpu.getData();
        float* logitsTRTFormat = static_cast<float*>(logitsUnformatted);
        size_t logitsCount = logitsCpu.getSize();
        constrainLogitsToGrammar(logitsTRTFormat, logitsCount, constrainingGrammar);
        
        logits.setFrom(logitsCpu, streamPtr);
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
    
    llama_grammar_free_impl(constrainingGrammar);

    // Get outputTokens
    auto outputTokens = responses.at(0).getResult().outputTokenIds.at(beamWidth - 1);

    TLLM_LOG_INFO("Output tokens: %s", tlc::vec2str(outputTokens).c_str());
    // }

    return 0;
}
