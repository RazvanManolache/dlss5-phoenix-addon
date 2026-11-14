// dlss5-feed IPC: the 32-bit in-game add-on <-> 64-bit host protocol.
//
// D3D11 shares NT textures and fences with a private D3D12 helper. Handles
// are fixed-width values, never pointers to objects in another process.
// The game submits one frame, then polls completion on a background worker.
// No game-queue Wait is recorded against a helper that might exit.
// Older graphics APIs use cpu_bridge and packed RGBA8 rows instead. A v14
// peer can negotiate a shared mapping for those rows and the optional depth.
// A v15 mapping is a ring of frame slots; each frame names its slot so the
// producer can stage the next frame while the helper processes the current.

#pragma once
#include <cstdint>
#include <cmath>

#define FEED_IPC_MAGIC   0x35534C44u  // 'DLS5'
#define FEED_IPC_VERSION 16u
#define FEED_IPC_MIN_VERSION 10u // Unchanged frame layout; older clients omit live tuning.
#define FEED_SESSION_PIPE_FMT "\\\\.\\pipe\\dlss5-feed.%lu.%llu" // game PID and per-session nonce
#define FEED_PIPE_FMT    "\\\\.\\pipe\\dlss5-feed.%lu"   // %lu = game PID
inline constexpr bool feed_protocol_compatible(uint32_t version) {
    return version >= FEED_IPC_MIN_VERSION && version <= FEED_IPC_VERSION;
}

enum FeedSlot { FEED_COLOR = 0, FEED_OUTPUT, FEED_DEPTH, FEED_MV, FEED_MASK, FEED_SLOTS };

#pragma pack(push, 1)

struct FeedHello        // game -> host, once
{
    uint32_t magic;
    uint32_t version;
    uint32_t pid;
};

struct FeedHelloAck     // host -> game, once
{
    uint32_t magic;
    uint32_t version;
};

struct FeedBuild        // game -> host, on every resolution/format change
{
    uint32_t width, height;      // render/input size fed to DLSS
    uint32_t output_width, output_height;
    uint32_t color_fmt;          // DXGI_FORMAT of the shared Color/Output pair
    uint32_t output_fmt;
    int32_t  hdr;                // resolved flags, not cfg values
    int32_t  depth_inverted;
    int32_t  flags_override;     // -1 = none
    int32_t  transport;          // 1 = no NGX: host copies Color -> Output (cross-process transport test)
    int32_t  has_mask;           // 1 = FEED_MASK contains DLSS5_Mask for pInBiasCurrentColorMask
    int32_t  cpu_bridge;         // 1 = frames arrive over the pipe as RGBA8 CPU rows (native D3D9/D3D12 path)
    float    mv_scale_x, mv_scale_y;
    int32_t  game_side_vr_split; // 1 = game sends one side-by-side VR eye per frame through shared textures
    uint32_t present_width, present_height;
    uint64_t adapter_luid;       // Render adapter; zero selects the default for CPU transport.
    uint64_t tex[FEED_SLOTS];    // NT-handle VALUES in the game process (host duplicates them out)
};

struct FeedBuildAck     // host -> game
{
    int32_t  ok;                 // 1 = feature ready
    uint32_t ngx_result;         // NVSDK_NGX_Result of the create (0x1 = success)
    uint64_t fence_in;           // handle values in the HOST process; client duplicates before opening
    uint64_t fence_out;
};

struct FeedFrameMsg     // game -> host, per frame
{
    uint64_t n;                  // fence value for this frame
    uint32_t reset;              // 1 = reset temporal history
    uint32_t nr_enabled;         // legacy field name: 1 = Phoenix (Backend=3), 0 = input bypass (Backend=0)
    uint32_t iterations;          // 1..10 evaluates of this frame before signaling output ready
    uint32_t eye_index;           // game_side_vr_split only: 0 = left eye, 1 = right eye
    uint32_t slot;               // v15: ring slot in the shared CPU mapping; pre-v15 size excludes it
};
inline constexpr size_t feed_frame_msg_bytes(uint32_t version) {
    return version >= 15 ? sizeof(FeedFrameMsg) : 24;
}

// Optional v13 'D' precedes its CPU 'F' packet. A native-size float plane
// follows the serial, using conventional near=0, far=1 device depth. v15
// packs the captured plane size into depth_dims as (width << 16) | height
// when it differs from the color size (0 = color size); the mapping depth
// plane then holds width*height floats at its start.
struct FeedCpuDepth { uint64_t frame; uint32_t slot; uint32_t depth_dims; };
inline constexpr uint32_t feed_pack_depth_dims(unsigned width, unsigned height) {
    return (uint32_t(width) << 16) | uint32_t(height);
}
inline constexpr unsigned feed_depth_width(uint32_t packed) { return packed >> 16; }
inline constexpr unsigned feed_depth_height(uint32_t packed) { return packed & 0xffffu; }
inline constexpr size_t feed_cpu_depth_bytes(uint32_t version) {
    return version >= 15 ? sizeof(FeedCpuDepth) : 8;
}

// Optional v14 'M' after an accepted CPU build. The host replies with a
// uint32_t acceptance flag. Both peers retain pipe transport if refused.
// v15 extends the request with the ring slot count (1 = the v14 layout).
// v16 can append per-slot prefetch metadata after the v15 pixel planes.
inline constexpr uint32_t FEED_CPU_PREFETCH = 1u;
struct FeedCpuMapping { uint64_t handle; uint32_t slots; uint32_t reserved; uint64_t wake_event; };
inline constexpr size_t feed_cpu_mapping_bytes(uint32_t version) {
    return version >= 16 ? sizeof(FeedCpuMapping) : version >= 15 ? 16 : 8;
}

struct FeedCpuFrameAck   // host -> game after a CPU-bridge frame
{
    int32_t  ok;                 // 1 = output pixels follow
    uint32_t row_bytes;          // output row bytes; currently output_width * 4
};

// Optional UI traffic runs on the pipe worker, separately from frame packets.
struct FeedUiAction { uint32_t action, value; };
struct FeedUiReply { uint32_t bytes; }; // Bounded UTF-8 JSON follows (at most 1 MiB).
struct FeedSettingsPatch { uint32_t bytes; }; // 'S': live-only JSON, <= 8192 bytes; uint32 acceptance reply.
// Optional v12 'G', immediately before one 'F'. A declaration applies to that
// frame only. The producer must copy real, history-aligned vectors into FEED_MV;
// allocated/zero-filled placeholder textures do not establish availability.
struct FeedGuideState { uint32_t engine_motion = 0; float scale_x = 1, scale_y = 1; };

#pragma pack(pop)

static_assert(sizeof(FeedBuild)==116);
static_assert(sizeof(FeedFrameMsg)==28);
static_assert(sizeof(FeedCpuDepth)==16);
static_assert(sizeof(FeedCpuMapping)==24);
static_assert(sizeof(FeedBuildAck)==24);
static_assert(sizeof(FeedGuideState)==12);

inline bool feed_guides_valid(const FeedGuideState& guides) {
    return guides.engine_motion <= 1 && std::isfinite(guides.scale_x) && std::isfinite(guides.scale_y);
}
// Consume each declaration once, so a previous frame's engine vectors can never
// silently turn an ordinary zero-filled transport texture into trusted motion.
inline FeedGuideState feed_take_guides(FeedGuideState& pending) {
    const auto current = pending; pending = {}; return current;
}
