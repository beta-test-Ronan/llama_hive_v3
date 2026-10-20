#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <clocale>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#ifdef HIVE_V3
enum class hive_task_kind : uint8_t { planner, critic, synthesizer };

struct hive_task_handoff {
    int source;
    int target;
    hive_task_kind kind;
    std::array<float, 128> payload;
};

static hive_task_kind hive_task_for_worker(int worker) {
    switch (worker % 3) {
        case 0: return hive_task_kind::planner;
        case 1: return hive_task_kind::critic;
        default: return hive_task_kind::synthesizer;
    }
}
#endif

#ifdef HIVE_TOPOLOGY_V2
static std::array<float, 128> hive_summary_init(int worker) {
    std::array<float, 128> s{};
    for (int d = 0; d < 128; ++d) {
        s[d] = std::sin(0.017f * float((worker + 1) * (d + 3)));
    }
    return s;
}

static void hive_summary_update(std::array<float, 128> & s, llama_token token, int step) {
    // A cheap deterministic 128-D sketch. It is a routing feature, not model hidden state.
    for (int d = 0; d < 128; ++d) {
        const float feature = std::sin(0.013f * float(token + 17 * d + 31 * step));
        s[d] = 0.96f * s[d] + 0.04f * feature;
    }
}

static std::vector<int> hive_topology_order(
        const std::vector<int> & workers,
        const std::vector<std::array<float, 128>> & summaries) {
    // Three fixed anchors form a lightweight triangulation of the worker space.
    std::array<std::array<float, 128>, 3> anchors{};
    for (int d = 0; d < 128; ++d) {
        anchors[0][d] = std::sin(0.071f * float(d + 1));
        anchors[1][d] = std::cos(0.053f * float(d + 5));
        anchors[2][d] = std::sin(0.037f * float(d + 11) + 1.7f);
    }

    struct route_key { int worker; int anchor; float distance; float projection; };
    std::vector<route_key> keys;
    keys.reserve(workers.size());
    for (int worker : workers) {
        int best_anchor = 0;
        float best_distance = INFINITY;
        for (int a = 0; a < 3; ++a) {
            float distance = 0.0f;
            for (int d = 0; d < 128; ++d) {
                const float delta = summaries[worker][d] - anchors[a][d];
                distance += delta * delta;
            }
            if (distance < best_distance) {
                best_distance = distance;
                best_anchor = a;
            }
        }
        float projection = 0.0f;
        for (int d = 0; d < 128; ++d) {
            projection += summaries[worker][d] * (0.5f + float((d % 7) + 1) / 8.0f);
        }
        keys.push_back({ worker, best_anchor, best_distance, projection });
    }
    std::stable_sort(keys.begin(), keys.end(), [](const route_key & a, const route_key & b) {
        if (a.anchor != b.anchor) return a.anchor < b.anchor;
        if (a.distance != b.distance) return a.distance < b.distance;
        return a.projection < b.projection;
    });
    std::vector<int> ordered;
    ordered.reserve(keys.size());
    for (const auto & key : keys) ordered.push_back(key.worker);
    return ordered;
}
#endif

#ifdef HIVE_KV_SELECT_V25
static constexpr int HIVE_V25_BLOCK_TOKENS = 16;
static constexpr int HIVE_V25_KEEP_BLOCKS = 8;
static constexpr int HIVE_V25_KEEP_RECENT = 2;

static float hive_block_score(
        const std::array<float, 128> & summary,
        const std::vector<llama_token> & history,
        int begin, int end) {
    std::array<float, 128> block{};
    for (int pos = begin; pos < end; ++pos) {
        const float token = float(history[pos]);
        for (int d = 0; d < 128; ++d) {
            block[d] += std::sin(0.009f * token + 0.017f * float(d + 1) + 0.003f * float(pos));
        }
    }
    float dot = 0.0f;
    float norm_s = 1e-6f;
    float norm_b = 1e-6f;
    for (int d = 0; d < 128; ++d) {
        dot += summary[d] * block[d];
        norm_s += summary[d] * summary[d];
        norm_b += block[d] * block[d];
    }
    return dot / std::sqrt(norm_s * norm_b);
}

static int hive_select_kv_blocks(
        llama_context * ctx,
        int worker,
        const std::vector<llama_token> & history,
        const std::array<float, 128> & summary) {
    const int n_blocks = (int(history.size()) + HIVE_V25_BLOCK_TOKENS - 1) / HIVE_V25_BLOCK_TOKENS;
    if (n_blocks <= HIVE_V25_KEEP_BLOCKS) return 0;

    struct scored_block { int index; float score; };
    std::vector<scored_block> candidates;
    const int first_recent = std::max(0, n_blocks - HIVE_V25_KEEP_RECENT);
    for (int block = 0; block < first_recent; ++block) {
        const int begin = block * HIVE_V25_BLOCK_TOKENS;
        const int end = std::min<int>(history.size(), begin + HIVE_V25_BLOCK_TOKENS);
        candidates.push_back({ block, hive_block_score(summary, history, begin, end) });
    }
    const int keep_old = HIVE_V25_KEEP_BLOCKS - HIVE_V25_KEEP_RECENT;
    const int keep_count = std::min<int>(keep_old, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + keep_count, candidates.end(),
        [](const scored_block & a, const scored_block & b) { return a.score > b.score; });

    std::vector<bool> keep(n_blocks, false);
    for (int block = first_recent; block < n_blocks; ++block) keep[block] = true;
    for (int i = 0; i < keep_count; ++i) keep[candidates[i].index] = true;

    llama_memory_t memory = llama_get_memory(ctx);
    int removed = 0;
    for (int block = 0; block < first_recent; ++block) {
        if (!keep[block]) {
            const llama_pos p0 = block * HIVE_V25_BLOCK_TOKENS;
            const llama_pos p1 = std::min<int>(history.size(), p0 + HIVE_V25_BLOCK_TOKENS);
            if (llama_memory_seq_rm(memory, worker, p0, p1)) ++removed;
        }
    }
    return removed;
}
#endif

static void print_usage(int, char ** argv) {
    LOG("\nexperimental usage:\n");
    LOG("\n    %s -m model.gguf -p \"Hello\" -n 32 -np 200 -c 4096 -ub 64\n", argv[0]);
    LOG("\nThis prototype uses one llama_context and up to 200 logical workers.\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "Hello";
    params.n_predict = 16;
    params.n_parallel = 200;
    params.n_ctx = 4096;
    params.n_batch = 512;
    params.n_ubatch = 64;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BATCHED, print_usage)) {
        return 1;
    }

    const int n_workers = std::clamp(params.n_parallel, 1, 200);
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (model == nullptr) {
        LOG_ERR("%s: unable to load model\n", __func__);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> prompt = common_tokenize(vocab, params.prompt, true);
    const int n_predict = std::max(1, params.n_predict);
    const int n_kv_req = prompt.size() + n_predict * n_workers;

    llama_context_params ctx_params = common_context_params_to_llama(params);
#ifdef HIVE_V3
    // 128 is a logical per-worker budget. Keep one unified cache so 200
    // workers do not allocate 200 independent physical KV lanes.
    ctx_params.n_ctx = std::max<int>(params.n_ctx, n_kv_req);
    ctx_params.kv_unified = true;
#else
    ctx_params.n_ctx = std::max<int>(params.n_ctx, n_kv_req);
#endif
    ctx_params.n_batch = std::max<uint32_t>(ctx_params.n_batch, n_workers);
    ctx_params.n_ubatch = std::min(ctx_params.n_ubatch, ctx_params.n_batch);
    ctx_params.n_seq_max = n_workers;
#ifndef HIVE_V3
    ctx_params.kv_unified = true;
#endif

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        LOG_ERR("%s: unable to create context\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    LOG_INF("hive: workers=%d, prompt_tokens=%zu, n_ctx=%u, n_batch=%u, n_ubatch=%u\n",
            n_workers, prompt.size(), llama_n_ctx(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx));
    LOG_INF("hive: one model/context, logical sequence lanes, shared prompt KV\n");
#ifdef HIVE_V3
    LOG_INF("hive-v3: one-token-per-tick, logical_lane_ctx=128, unified_physical_ctx=%u\n", llama_n_ctx(ctx));
#endif
#ifdef HIVE_TOPOLOGY_V2
    LOG_INF("hive-v2: 128-D sketch + 3-anchor triangulation orders each decode microbatch\n");
#else
    LOG_INF("hive-v1: full batch baseline; use -ctk q8_0 -ctv q8_0 for quantized KV\n");
#endif
#ifdef HIVE_KV_SELECT_V25
    LOG_INF("hive-v2.5: score KV blocks with 128-D similarity; keep=%d recent=%d block_tokens=%d\n",
            HIVE_V25_KEEP_BLOCKS, HIVE_V25_KEEP_RECENT, HIVE_V25_BLOCK_TOKENS);
#endif

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    std::vector<llama_sampler *> samplers;
    samplers.reserve(n_workers);
    for (int i = 0; i < n_workers; ++i) {
        llama_sampler * sampler = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234 + i));
        samplers.push_back(sampler);
    }

    common_batch batch(ctx);
    std::vector<llama_seq_id> seq_ids(n_workers);
    for (int i = 0; i < n_workers; ++i) {
        seq_ids[i] = i;
    }

    for (size_t pos = 0; pos < prompt.size(); ++pos) {
        // All workers consume the same prompt. The KV implementation can share it.
        batch.add(prompt[pos], pos, seq_ids, false);
    }
    batch.set_output(batch.size() - 1, true);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("hive: prompt evaluation failed\n");
        for (auto * sampler : samplers) llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::vector<int32_t> output_index(n_workers, batch.size() - 1);
    std::vector<int> generated(n_workers, 0);
#ifdef HIVE_TOPOLOGY_V2
    std::vector<std::array<float, 128>> summaries(n_workers);
    for (int i = 0; i < n_workers; ++i) summaries[i] = hive_summary_init(i);
#endif
#ifdef HIVE_KV_SELECT_V25
    std::vector<std::vector<llama_token>> histories(n_workers, prompt);
    int total_removed_blocks = 0;
#endif
#ifdef HIVE_V3
    std::vector<hive_task_handoff> task_queue;
    task_queue.reserve(n_workers);
    int64_t handoffs = 0;
    std::string final_answer;
    const int final_worker = n_workers - 1;
#endif
    int active = n_workers;
    int64_t total_tokens = 0;
    const int64_t t0 = ggml_time_us();

    while (active > 0 && total_tokens < int64_t(n_predict) * n_workers) {
        batch.clear();
        std::vector<int> ready;
        std::vector<llama_token> next_tokens(n_workers, LLAMA_TOKEN_NULL);
        for (int i = 0; i < n_workers; ++i) {
            if (generated[i] >= n_predict) continue;
            const llama_token token = llama_sampler_sample(samplers[i], ctx, output_index[i]);
            llama_sampler_accept(samplers[i], token);
            if (llama_vocab_is_eog(vocab, token)) {
                generated[i] = n_predict;
                --active;
                continue;
            }
            next_tokens[i] = token;
            ready.push_back(i);
#ifdef HIVE_V3
            if (i == final_worker) {
                final_answer += common_token_to_piece(ctx, token);
            }
#endif
#ifdef HIVE_V3
            hive_task_handoff task{};
            task.source = i;
            task.target = (i + 1) % n_workers;
            task.kind = hive_task_for_worker(task.target);
            task.payload = summaries[i];
            task_queue.push_back(task);
            ++handoffs;
#endif
        }
#ifdef HIVE_V3
        for (const auto & task : task_queue) {
            // Metadata handoff only: no extra prompt tokens are added.
            for (int d = 0; d < 128; ++d) {
                summaries[task.target][d] = 0.9f * summaries[task.target][d] + 0.1f * task.payload[d];
            }
        }
        task_queue.clear();
#endif
#ifdef HIVE_TOPOLOGY_V2
        ready = hive_topology_order(ready, summaries);
#endif
        for (int i : ready) {
            const llama_token token = next_tokens[i];
            const int32_t row = batch.size();
            batch.add(token, prompt.size() + generated[i], i, true);
            output_index[i] = row;
#ifdef HIVE_TOPOLOGY_V2
            hive_summary_update(summaries[i], token, generated[i]);
#endif
#ifdef HIVE_KV_SELECT_V25
            histories[i].push_back(token);
#endif
            ++generated[i];
            ++total_tokens;
        }
        if (batch.size() == 0) break;
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("hive: decode failed after %lld tokens\n", (long long) total_tokens);
            break;
        }
#ifdef HIVE_KV_SELECT_V25
        if (!ready.empty() && generated[ready.front()] % HIVE_V25_BLOCK_TOKENS == 0) {
            llama_synchronize(ctx);
            for (int worker : ready) {
                total_removed_blocks += hive_select_kv_blocks(ctx, worker, histories[worker], summaries[worker]);
            }
        }
#endif
    }

    const int64_t t1 = ggml_time_us();
    const double seconds = (t1 - t0) / 1000000.0;
    LOG_INF("hive: generated %lld tokens in %.3f s (%.2f tok/s)\n",
            (long long) total_tokens, seconds, total_tokens / std::max(seconds, 1e-9));
    LOG_INF("hive: active workers at end: %d/%d\n", active, n_workers);
#ifdef HIVE_V3
    LOG_INF("hive-v3: task handoffs=%lld, roles=planner/critic/synthesizer\n", (long long) handoffs);
    while (!final_answer.empty() && std::isspace((unsigned char) final_answer.back())) final_answer.pop_back();
    if (!final_answer.empty() && final_answer.back() != '.' && final_answer.back() != '!' && final_answer.back() != '?') {
        final_answer.push_back('.');
    }
    LOG("[Worker %d / final]: %s\n", final_worker, final_answer.c_str());
#endif
#ifdef HIVE_KV_SELECT_V25
    LOG_INF("hive-v2.5: removed KV blocks=%d\n", total_removed_blocks);
#endif
    llama_perf_context_print(ctx);

    for (auto * sampler : samplers) llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
