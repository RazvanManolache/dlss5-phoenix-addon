#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace phoenix_addon {
inline void check(bool value, const char* reason) {
    if (!value)
        throw std::runtime_error(reason);
}

// Dispatch scheduling: one NvAPI chained launch per natural group (15
// calls) instead of one per kernel (154 calls). Same kernels, same
// order, same captured barriers -- only the call grouping differs, so
// output must be bit-identical; the group profiler host runs this shape
// with a per-frame parity gate. CNR_PHOENIX_BATCH=group selects it; anything
// else, including absent, keeps per-kernel dispatch. CPU tests pin it.
inline bool parse_batch_group(const char* text) noexcept {
    if (!text || !*text)
        return false;
    return std::strcmp(text, "group") == 0;
}

// Cubin selection: CNR_PHOENIX_CUBINS=native loads the DLL's own kernel
// modules instead of the reconstructed cubins (same executor, plans,
// weights). Anything else, including absent, runs candidate cubins.
// CPU tests pin it.
inline bool parse_native_cubins(const char* text) noexcept {
    if (!text || !*text)
        return false;
    return std::strcmp(text, "native") == 0;
}

// Chain-observation accounting (pure): kernels/chains launched inside one
// Evaluate, attributed by the live addon's NvAPI detours. Insane counts are
// ignored and the total saturates: the counter must never wrap, or a busy
// frame would look like an idle one.
struct ChainAccount {
    uint64_t kernels = 0;
    uint64_t chains = 0;
    void add(uint64_t count) noexcept {
        if (count == 0 || count >= 1000000u)
            return;
        chains += 1;
        kernels = (kernels + count < kernels) ? UINT64_MAX : kernels + count;
    }
    void reset() noexcept {
        kernels = 0;
        chains = 0;
    }
};

// Ours-driver budget (pure): integer EMA (alpha 1/8) of per-call cost, and
// the skip decision. Budget 0 = off: never skip.
inline uint64_t budget_ema_update(uint64_t prev_ema, uint64_t sample_us) noexcept {
    return (prev_ema * 7 + sample_us) / 8;
}
inline bool budget_should_skip(uint64_t ema_us, int budget_us) noexcept {
    return budget_us > 0 && ema_us > static_cast<uint64_t>(budget_us);
}

// Ablation (pure): bit i skips the i-th chain launch inside one Evaluate.
// Positional hypothesis (one chained launch per natural group, plan order):
// the live tab shows chains/call so the mapping is checkable per game.
inline bool ablation_skips_chain(uint32_t mask, unsigned chain_index) noexcept {
    if (chain_index >= 32)
        return false;
    return ((mask >> chain_index) & 1u) != 0;
}

// Ours model-file check (pure): a loadable kernel module must at least look
// like an ELF/cubin (7f 'E' 'L' 'F') and clear the ELF header size. The live
// addon polls the configured path and reports readiness; weights arrive when
// the user points this at a real cubin.
inline bool looks_like_cubin(const unsigned char* bytes, size_t size) noexcept {
    if (!bytes || size < 64)
        return false;
    return bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F';
}

// Natural dispatch groups (15) and their cumulative launch ends: 154
// launches total, positions 0..153 in plan order. Shared source of truth
// for the skip mask; the group profiler pins the same schedule at plan
// load and Session construction requires it, so a schedule change fails
// loud instead of mis-skipping. CPU tests pin the constants.
constexpr unsigned kGroupCount = 15;
constexpr unsigned kLaunchCount = 154;
constexpr unsigned kGroupEnds[kGroupCount] = {1,  5,   9,   15,  23,  56,  57, 97,
                                              98, 131, 139, 145, 149, 153, 154};
constexpr const char* kGroupLabels[kGroupCount] = {"stem",
                                                   "encoder_32",
                                                   "encoder_64",
                                                   "encoder_128",
                                                   "encoder_256",
                                                   "encoder_split_512",
                                                   "repack_2d_to_1d",
                                                   "vit_1024",
                                                   "repack_1d_to_2d",
                                                   "decoder_split_512",
                                                   "decoder_256",
                                                   "decoder_128",
                                                   "decoder_64",
                                                   "decoder_32",
                                                   "head"};

// Ablation/execution control: bit i skips natural group i. A skipped
// group launches nothing -- its tensors stay stale, history still
// advances, barriers and group timestamps are still recorded, so a
// skipped frame stays comparable. Written by the host (overlay, env,
// config), snapshotted once per recorded frame.
inline std::atomic<uint32_t> skip_groups_mask{0};

inline unsigned group_of_position(unsigned position) {
    check(position < kLaunchCount, "phoenix launch position outside frame");
    for (unsigned group = 0; group < kGroupCount; ++group)
        if (position < kGroupEnds[group])
            return group;
    check(false, "phoenix launch position outside frame");
    return 0; // Unreachable: the last end equals the launch count.
}

inline unsigned skipped_launches(uint32_t mask) noexcept {
    unsigned skipped = 0, begin = 0;
    for (unsigned group = 0; group < kGroupCount; ++group) {
        if (mask & (1u << group))
            skipped += kGroupEnds[group] - begin;
        begin = kGroupEnds[group];
    }
    return skipped;
}

// Parses a comma-separated group list: labels ("vit_1024") or indices
// ("7"), e.g. "encoder_128,decoder_128" or "3,11". Empty/absent selects
// nothing. Unknown tokens throw: a mistyped ablation must fail loud,
// never silently run the wrong network. CPU tests pin it.
inline uint32_t parse_skip_groups(const char* text) {
    if (!text || !*text)
        return 0;
    uint32_t mask = 0;
    const char* token = text;
    for (;;) {
        const char* end = token;
        while (*end && *end != ',')
            ++end;
        check(end != token, "phoenix empty skip group token");
        unsigned index = 0;
        bool numeric = true;
        for (const char* token_cursor = token; token_cursor != end; ++token_cursor) {
            if (*token_cursor < '0' || *token_cursor > '9') {
                numeric = false;
                break;
            }
            index = index * 10 + unsigned(*token_cursor - '0');
        }
        if (numeric) {
            check(index < kGroupCount, "phoenix skip group index out of range");
            mask |= 1u << index;
        } else {
            bool known = false;
            for (unsigned group = 0; group < kGroupCount; ++group) {
                const char* label = kGroupLabels[group];
                const char* token_cursor = token;
                const char* label_cursor = label;
                while (token_cursor != end && *label_cursor && *token_cursor == *label_cursor) {
                    ++token_cursor;
                    ++label_cursor;
                }
                if (token_cursor == end && *label_cursor == 0) {
                    mask |= 1u << group;
                    known = true;
                    break;
                }
            }
            check(known, "phoenix unknown skip group");
        }
        if (!*end)
            return mask;
        token = end + 1;
    }
}

// Independent of D3D12 so failure and abandonment behavior can be tested on CPU.
//
// Recording stays single threaded: only the thread that created the session may
// record into a command list. Submission does not. A measured Unreal D3D12
// frame records the NR boundary on one thread and calls ExecuteCommandLists on
// another, so the submission contract is identity based - exact command list,
// exact queue, still pending, not failed - and the caller serializes recording
// against submission with its own lock.
class Transaction {
    std::thread::id recording_ = std::this_thread::get_id();
    std::thread::id submission_{};
    uintptr_t command_ = 0, queue_ = 0;
    bool failed_ = false, pending_ = false;
    bool cross_thread_submission_ = false;
    uint64_t committed_ = 0;

  public:
    void validate(uintptr_t command, uintptr_t queue) const {
        check(recording_ == std::this_thread::get_id(), "phoenix game wrong recording thread");
        check(!failed_ && !pending_ && command && queue,
              "phoenix game failed/pending/invalid transaction");
        check(!queue_ || queue_ == queue, "phoenix game queue change requires a new session");
    }
    void begin(uintptr_t command, uintptr_t queue) {
        validate(command, queue);
        command_ = command;
        queue_ = queue;
        pending_ = true;
    }
    bool submission_matches(uintptr_t command, uintptr_t queue) const noexcept {
        return pending_ && !failed_ && command == command_ && queue == queue_;
    }
    void submitted(uintptr_t command, uintptr_t queue, bool success) {
        if (!success || !pending_ || failed_ || command != command_ || queue != queue_) {
            invalidate();
            throw std::runtime_error("phoenix game submission did not satisfy commit contract");
        }
        submission_ = std::this_thread::get_id();
        if (submission_ != recording_)
            cross_thread_submission_ = true;
        pending_ = false;
        command_ = 0;
        ++committed_;
    }
    void invalidate() noexcept {
        failed_ = true;
        pending_ = false;
        command_ = 0;
    }
    bool pending() const noexcept {
        return pending_;
    }
    bool failed() const noexcept {
        return failed_;
    }
    // Identity evidence for the submission gate: which command/queue the
    // open transaction binds, so a block names the mismatching side.
    uintptr_t pending_command() const noexcept {
        return command_;
    }
    uintptr_t pending_queue() const noexcept {
        return queue_;
    }
    uint64_t committed() const noexcept {
        return committed_;
    }
    // True only on the recording thread. Recording, committing the reconstructed
    // network and destroying owned GPU objects all require it.
    bool owner_thread() const noexcept {
        return recording_ == std::this_thread::get_id();
    }
    bool cross_thread_submission() const noexcept {
        return cross_thread_submission_;
    }
};
} // namespace production_game41
