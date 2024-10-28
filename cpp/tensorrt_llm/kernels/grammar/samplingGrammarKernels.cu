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

// NOTE: assumes valid utf8 (but checks for overrun)
static std::pair<uint32_t, const char *> decode_utf8(const char * src) {
    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t  first_byte = static_cast<uint8_t>(*src);
    uint8_t  highbits   = first_byte >> 4;
    int      len        = lookup[highbits];
    uint8_t  mask       = (1 << (8 - len)) - 1;
    uint32_t value      = first_byte & mask;
    const char * end    = src + len; // may overrun!
    const char * pos    = src + 1;
    for ( ; pos < end && *pos; pos++) {
        value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
    }
    return std::make_pair(value, pos);
}

__device__ __host__ static std::pair<std::vector<uint32_t>, llama_partial_utf8> decode_utf8(
    const std::string & src,
    llama_partial_utf8 partial_start) {
    static const int      lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
    const char          * pos      = src.c_str();
    std::vector<uint32_t> code_points;

    // common English strings have the same number of codepoints and bytes. `+ 1` for the terminating 0.
    code_points.reserve(src.size() + 1);
    uint32_t value    = partial_start.value;
    int      n_remain = partial_start.n_remain;

    // Continue previous decode, if applicable
    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            // Invalid sequence, abort
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{ 0, -1 });
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }

    if (partial_start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    // Decode any subsequent utf-8 sequences, which may end in an incomplete one
    while (*pos != 0) {
        uint8_t first_byte = static_cast<uint8_t>(*pos);
        uint8_t highbits   = first_byte >> 4;
        n_remain   = lookup[highbits] - 1;

        if (n_remain < 0) {
            // Invalid sequence, abort
            code_points.clear();
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{ 0, n_remain });
        }

        uint8_t mask  = (1 << (7 - n_remain)) - 1;
        value = first_byte & mask;

        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) {
            code_points.push_back(value);
        }
    }
    code_points.push_back(0);

    return std::make_pair(std::move(code_points), llama_partial_utf8{ value, n_remain });
}

__device__ __host__ static bool is_digit_char(char c) {
    return '0' <= c && c <= '9';
}

__device__ __host__ static bool is_word_char(char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || is_digit_char(c);
}

__device__ __host__ static std::pair<uint32_t, const char *> parse_hex(const char * src, int size) {
    const char * pos   = src;
    const char * end   = src + size;
    uint32_t     value = 0;
    for (; pos < end && *pos; pos++) {
        value <<= 4;
        char c = *pos;
        if ('a' <= c && c <= 'f') {
            value += c - 'a' + 10;
        } else if ('A' <= c && c <= 'F') {
            value += c - 'A' + 10;
        } else if ('0' <= c && c <= '9') {
            value += c - '0';
        } else {
            break;
        }
    }
    if (pos != end) {
        throw std::runtime_error("expecting " + std::to_string(size) + " hex chars at " + std::string(src));
    }
    return std::make_pair(value, pos);
}

__device__ __host__ static const char * parse_space(const char * src, bool newline_ok) {
    const char * pos = src;
    while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
        (newline_ok && (*pos == '\r' || *pos == '\n'))) {
        if (*pos == '#') {
            while (*pos && *pos != '\r' && *pos != '\n') {
                pos++;
            }
        } else {
            pos++;
        }
    }
    return pos;
}

__device__ __host__ static const char * parse_name(const char * src) {
    const char * pos = src;
    while (is_word_char(*pos)) {
        pos++;
    }
    if (pos == src) {
        throw std::runtime_error(std::string("expecting name at ") + src);
    }
    return pos;
}

__device__ __host__ static const char * parse_int(const char * src) {
    const char * pos = src;
    while (is_digit_char(*pos)) {
        pos++;
    }
    if (pos == src) {
        throw std::runtime_error(std::string("expecting integer at ") + src);
    }
    return pos;
}

__device__ __host__ static std::pair<uint32_t, const char *> parse_char(const char * src) {
    if (*src == '\\') {
        switch (src[1]) {
            case 'x': return parse_hex(src + 2, 2);
            case 'u': return parse_hex(src + 2, 4);
            case 'U': return parse_hex(src + 2, 8);
            case 't': return std::make_pair('\t', src + 2);
            case 'r': return std::make_pair('\r', src + 2);
            case 'n': return std::make_pair('\n', src + 2);
            case '\\':
            case '"':
            case '[':
            case ']':
                return std::make_pair(src[1], src + 2);
            default:
                throw std::runtime_error(std::string("unknown escape at ") + src);
        }
    } else if (*src) {
        return decode_utf8(src);
    }
    throw std::runtime_error("unexpected end of input");
}

__device__ __host__ static void print_grammar_char(FILE * file, uint32_t c) {
    if (0x20 <= c && c <= 0x7f) {
        fprintf(file, "%c", static_cast<char>(c));
    } else {
        // cop out of encoding UTF-8
        fprintf(file, "<U+%04X>", c);
    }
}

__device__ __host__ static bool is_char_element(llama_grammar_element elem) {
    switch (elem.type) {
        case LLAMA_GRETYPE_CHAR:           return true;
        case LLAMA_GRETYPE_CHAR_NOT:       return true;
        case LLAMA_GRETYPE_CHAR_ALT:       return true;
        case LLAMA_GRETYPE_CHAR_RNG_UPPER: return true;
        case LLAMA_GRETYPE_CHAR_ANY:       return true;
        default:                           return false;
    }
}

__device__ __host__ static void print_rule_binary(FILE * file, const llama_grammar_rule & rule) {
    for (auto elem : rule) {
        switch (elem.type) {
            case LLAMA_GRETYPE_END:            fprintf(file, "END");            break;
            case LLAMA_GRETYPE_ALT:            fprintf(file, "ALT");            break;
            case LLAMA_GRETYPE_RULE_REF:       fprintf(file, "RULE_REF");       break;
            case LLAMA_GRETYPE_CHAR:           fprintf(file, "CHAR");           break;
            case LLAMA_GRETYPE_CHAR_NOT:       fprintf(file, "CHAR_NOT");       break;
            case LLAMA_GRETYPE_CHAR_RNG_UPPER: fprintf(file, "CHAR_RNG_UPPER"); break;
            case LLAMA_GRETYPE_CHAR_ALT:       fprintf(file, "CHAR_ALT");       break;
            case LLAMA_GRETYPE_CHAR_ANY:       fprintf(file, "CHAR_ANY");       break;
        }
        switch (elem.type) {
            case LLAMA_GRETYPE_END:
            case LLAMA_GRETYPE_ALT:
            case LLAMA_GRETYPE_RULE_REF:
                fprintf(file, "(%u) ", elem.value);
                break;
            case LLAMA_GRETYPE_CHAR:
            case LLAMA_GRETYPE_CHAR_NOT:
            case LLAMA_GRETYPE_CHAR_RNG_UPPER:
            case LLAMA_GRETYPE_CHAR_ALT:
            case LLAMA_GRETYPE_CHAR_ANY:
                fprintf(file, "(\"");
                print_grammar_char(file, elem.value);
                fprintf(file, "\") ");
                break;
        }
    }
    fprintf(file, "\n");
}

__device__ __host__ static void print_rule(
    FILE     * file,
    uint32_t   rule_id,
    const llama_grammar_rule & rule,
    const std::map<uint32_t, std::string> & symbol_id_names) {
    if (rule.empty() || rule.back().type != LLAMA_GRETYPE_END) {
        throw std::runtime_error(
            "malformed rule, does not end with LLAMA_GRETYPE_END: " + std::to_string(rule_id));
    }
    fprintf(file, "%s ::= ", symbol_id_names.at(rule_id).c_str());
    for (size_t i = 0, end = rule.size() - 1; i < end; i++) {
        llama_grammar_element elem = rule[i];
        switch (elem.type) {
            case LLAMA_GRETYPE_END:
                throw std::runtime_error(
                    "unexpected end of rule: " + std::to_string(rule_id) + "," +
                    std::to_string(i));
            case LLAMA_GRETYPE_ALT:
                fprintf(file, "| ");
                break;
            case LLAMA_GRETYPE_RULE_REF:
                fprintf(file, "%s ", symbol_id_names.at(elem.value).c_str());
                break;
            case LLAMA_GRETYPE_CHAR:
                fprintf(file, "[");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_NOT:
                fprintf(file, "[^");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_RNG_UPPER:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                fprintf(file, "-");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_ALT:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "LLAMA_GRETYPE_CHAR_ALT without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_ANY:
                fprintf(file, ".");
                break;
        }
        if (is_char_element(elem)) {
            switch (rule[i + 1].type) {
                case LLAMA_GRETYPE_CHAR_ALT:
                case LLAMA_GRETYPE_CHAR_RNG_UPPER:
                case LLAMA_GRETYPE_CHAR_ANY:
                    break;
                default:
                    fprintf(file, "] ");
            }
        }
    }
    fprintf(file, "\n");
}

//
// Implementation
//

// Note: CUDA device code cannot use exceptions. We'll handle errors via return codes and status variables.

// Adjusted class definitions and methods to be CUDA-compatible.

struct llama_grammar_parser {
    std::map<std::string, uint32_t> symbol_ids;
    std::vector<llama_grammar_rule> rules;

    __device__ __host__ uint32_t get_symbol_id(const char * src, size_t len);
    __device__ __host__ uint32_t generate_symbol_id(const std::string & base_name);
    __device__ __host__ void add_rule(uint32_t rule_id, const llama_grammar_rule & rule);
    __device__ __host__ const char * parse_alternates(
        const char        * src,
        const std::string & rule_name,
        uint32_t            rule_id,
        bool                is_nested);
    __device__ __host__ const char * parse_sequence(
        const char         * src,
        const std::string  & rule_name,
        llama_grammar_rule & rule,
        bool                is_nested);
    __device__ __host__ const char * parse_rule(const char * src);
    __device__ __host__ bool parse(const char * src);
    __device__ __host__ void print(FILE * file);
    __device__ __host__ llama_grammar_stack c_rules() const;
};

__device__ __host__ uint32_t llama_grammar_parser::get_symbol_id(const char * src, size_t len) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    auto result = symbol_ids.emplace(std::string(src, len), next_id);
    if (!result.second) {
        next_id = result.first->second;
    }
    return next_id;
}

__device__ __host__ uint32_t llama_grammar_parser::generate_symbol_id(const std::string & base_name) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    std::string new_name = base_name + '_' + std::to_string(next_id);
    symbol_ids[new_name] = next_id;
    return next_id;
}

__device__ __host__ void llama_grammar_parser::add_rule(uint32_t rule_id, const llama_grammar_rule & rule) {
    if (rules.size() <= rule_id) {
        rules.resize(rule_id + 1);
    }
    rules[rule_id] = rule;
}

__device__ __host__ const char * llama_grammar_parser::parse_alternates(
    const char        * src,
    const std::string & rule_name,
    uint32_t            rule_id,
    bool                is_nested) {
    llama_grammar_rule rule;
    const char * pos = parse_sequence(src, rule_name, rule, is_nested);
    while (*pos == '|') {
        rule.push_back({ LLAMA_GRETYPE_ALT, 0 });
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(pos, rule_name, rule, is_nested);
    }
    rule.push_back({ LLAMA_GRETYPE_END, 0 });
    add_rule(rule_id, rule);
    return pos;
}

__device__ __host__ const char * llama_grammar_parser::parse_sequence(
    const char         * src,
    const std::string  & rule_name,
    llama_grammar_rule & rule,
    bool               is_nested) {
    size_t last_sym_start = rule.size();
    const char * pos = src;

    auto handle_repetitions = [&](int min_times, int max_times) {

        if (last_sym_start == rule.size()) {
            throw std::runtime_error(std::string("expecting preceding item to */+/?/{ at ") + pos);
        }

        llama_grammar_rule prev_rule(rule.begin() + last_sym_start, rule.end());
        if (min_times == 0) {
            rule.resize(last_sym_start);
        } else {
            // Repeat the previous elements (min_times - 1) times
            for (int i = 1; i < min_times; i++) {
                rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
            }
        }

        uint32_t last_rec_rule_id = 0;
        auto n_opt = max_times < 0 ? 1 : max_times - min_times;

        llama_grammar_rule rec_rule(prev_rule);
        for (int i = 0; i < n_opt; i++) {
            rec_rule.resize(prev_rule.size());
            uint32_t rec_rule_id = generate_symbol_id(rule_name);
            if (i > 0 || max_times < 0) {
                rec_rule.push_back({ LLAMA_GRETYPE_RULE_REF, max_times < 0 ? rec_rule_id : last_rec_rule_id });
            }
            rec_rule.push_back({ LLAMA_GRETYPE_ALT, 0 });
            rec_rule.push_back({ LLAMA_GRETYPE_END, 0 });
            add_rule(rec_rule_id, rec_rule);
            last_rec_rule_id = rec_rule_id;
        }
        if (n_opt > 0) {
            rule.push_back({ LLAMA_GRETYPE_RULE_REF, last_rec_rule_id });
        }
    };

    while (*pos) {
        if (*pos == '"') { // literal string
            pos++;
            last_sym_start = rule.size();
            while (*pos != '"') {
                if (!*pos) {
                    throw std::runtime_error("unexpected end of input");
                }
                auto char_pair = parse_char(pos);
                pos = char_pair.second;
                rule.push_back({ LLAMA_GRETYPE_CHAR, char_pair.first });
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '[') { // char range(s)
            pos++;
            enum llama_gretype start_type = LLAMA_GRETYPE_CHAR;
            if (*pos == '^') {
                pos++;
                start_type = LLAMA_GRETYPE_CHAR_NOT;
            }
            last_sym_start = rule.size();
            while (*pos != ']') {
                if (!*pos) {
                    throw std::runtime_error("unexpected end of input");
                }
                auto char_pair = parse_char(pos);
                pos = char_pair.second;
                enum llama_gretype type = last_sym_start < rule.size()
                    ? LLAMA_GRETYPE_CHAR_ALT
                    : start_type;

                rule.push_back({ type, char_pair.first });
                if (pos[0] == '-' && pos[1] != ']') {
                    if (!pos[1]) {
                        throw std::runtime_error("unexpected end of input");
                    }
                    auto endchar_pair = parse_char(pos + 1);
                    pos = endchar_pair.second;
                    rule.push_back({ LLAMA_GRETYPE_CHAR_RNG_UPPER, endchar_pair.first });
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (is_word_char(*pos)) { // rule reference
            const char * name_end = parse_name(pos);
            uint32_t ref_rule_id = get_symbol_id(pos, name_end - pos);
            pos = parse_space(name_end, is_nested);
            last_sym_start = rule.size();
            rule.push_back({ LLAMA_GRETYPE_RULE_REF, ref_rule_id });
        } else if (*pos == '(') { // grouping
            // parse nested alternates into synthesized rule
            pos = parse_space(pos + 1, true);
            uint32_t sub_rule_id = generate_symbol_id(rule_name);
            pos = parse_alternates(pos, rule_name, sub_rule_id, true);
            last_sym_start = rule.size();
            // output reference to synthesized rule
            rule.push_back({ LLAMA_GRETYPE_RULE_REF, sub_rule_id });
            if (*pos != ')') {
                throw std::runtime_error(std::string("expecting ')' at ") + pos);
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '.') { // any char
            last_sym_start = rule.size();
            rule.push_back({ LLAMA_GRETYPE_CHAR_ANY, 0 });
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '*') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, -1);
        } else if (*pos == '+') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(1, -1);
        } else if (*pos == '?') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, 1);
        } else if (*pos == '{') {
            pos = parse_space(pos + 1, is_nested);

            if (!is_digit_char(*pos)) {
                throw std::runtime_error(std::string("expecting an int at ") + pos);
            }
            const char * int_end = parse_int(pos);
            int min_times = std::stoul(std::string(pos, int_end - pos));
            pos = parse_space(int_end, is_nested);

            int max_times = -1;

            if (*pos == '}') {
                max_times = min_times;
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == ',') {
                pos = parse_space(pos + 1, is_nested);

                if (is_digit_char(*pos)) {
                    const char * int_end = parse_int(pos);
                    max_times = std::stoul(std::string(pos, int_end - pos));
                    pos = parse_space(int_end, is_nested);
                }

                if (*pos != '}') {
                    throw std::runtime_error(std::string("expecting '}' at ") + pos);
                }
                pos = parse_space(pos + 1, is_nested);
            } else {
                throw std::runtime_error(std::string("expecting ',' at ") + pos);
            }
            handle_repetitions(min_times, max_times);
        } else {
            break;
        }
    }
    return pos;
}

__device__ __host__ const char * llama_grammar_parser::parse_rule(const char * src) {
    const char * name_end = parse_name(src);
    const char * pos = parse_space(name_end, false);
    size_t       name_len = name_end - src;
    uint32_t     rule_id = get_symbol_id(src, name_len);
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error(std::string("expecting ::= at ") + pos);
    }
    pos = parse_space(pos + 3, true);

    pos = parse_alternates(pos, name, rule_id, false);

    if (*pos == '\r') {
        pos += pos[1] == '\n' ? 2 : 1;
    } else if (*pos == '\n') {
        pos++;
    } else if (*pos) {
        throw std::runtime_error(std::string("expecting newline or end at ") + pos);
    }
    return parse_space(pos, true);
}

__device__ __host__ bool llama_grammar_parser::parse(const char * src) {
    try {
        const char * pos = parse_space(src, true);
        while (*pos) {
            pos = parse_rule(pos);
        }
        // Validate the state to ensure that all rules are defined
        for (const auto & rule : rules) {
            if (rule.empty()) {
                throw std::runtime_error("Undefined rule");
            }
            for (const auto & elem : rule) {
                if (elem.type == LLAMA_GRETYPE_RULE_REF) {
                    // Ensure that the rule at that location exists
                    if (elem.value >= rules.size() || rules[elem.value].empty()) {
                        // Get the name of the rule that is missing
                        for (const auto & kv : symbol_ids) {
                            if (kv.second == elem.value) {
                                throw std::runtime_error("Undefined rule identifier '" + kv.first + "'");
                            }
                        }
                    }
                }
            }
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "Error parsing grammar: %s\n", err.what());
        rules.clear();
        return false;
    }

    return true;
}

__device__ __host__ void llama_grammar_parser::print(FILE * file) {
    try {
        std::map<uint32_t, std::string> symbol_id_names;
        for (const auto & kv : symbol_ids) {
            symbol_id_names[kv.second] = kv.first;
        }
        for (size_t i = 0, end = rules.size(); i < end; i++) {
            print_rule(file, uint32_t(i), rules[i], symbol_id_names);
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "Error printing grammar: %s\n", err.what());
    }
}

__device__ __host__ llama_grammar_stack llama_grammar_parser::c_rules() const {
    llama_grammar_stack ret;
    ret.reserve(rules.size());
    for (const auto & rule : rules) {
        ret.push_back(rule.data());
    }
    return ret;
}

template <typename T>
__global__ void applyCFGConstraintsKernel(GrammarSamplingKernelParams<T> params)
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
void invokeApplyCFGConstraints(GrammarSamplingKernelParams<T>& params, cudaStream_t stream)
{
    int32_t threadsPerBlock = 256;
    int32_t blocksPerGrid = params.batchSize;

    applyCFGConstraintsKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(params);

    // Check for errors
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void invokeBatchGrammarSampling(SamplingKernelParams<T>& params, cudaStream_t stream)
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
template void invokeApplyCFGConstraints<float>(GrammarSamplingKernelParams<float>& params, cudaStream_t stream);
template void invokeApplyCFGConstraints<half>(GrammarSamplingKernelParams<half>& params, cudaStream_t stream);

template void invokeBatchGrammarSampling<float>(SamplingKernelParams<float>& params, cudaStream_t stream);
template void invokeBatchGrammarSampling<half>(SamplingKernelParams<half>& params, cudaStream_t stream);

} // namespace tensorrt_llm::kernels

#undef CHECK_CUDA
