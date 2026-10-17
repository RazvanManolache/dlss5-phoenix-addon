// dlss5-feed IPC: the 32-bit in-game add-on <-> 64-bit host protocol.
//
// D3D11 shares NT textures and fences with a private D3D12 helper. Handles
// are fixed-width values, never pointers to objects in another process.
// The game submits one frame, then polls completion on a background worker.
// No game-queue Wait is recorded against a helper that might exit.
// Older graphics APIs use cpu_bridge and packed RGBA8 rows instead.

#pragma once
#include <cstdint>

#define FEED_IPC_MAGIC   0x35534C44u  // 'DLS5'
#define FEED_IPC_VERSION 11u
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
};

struct FeedCpuFrameAck   // host -> game after a CPU-bridge frame
{
    int32_t  ok;                 // 1 = output pixels follow
    uint32_t row_bytes;          // output row bytes; currently output_width * 4
};

// Optional UI traffic runs on the pipe worker, separately from frame packets.
struct FeedUiAction { uint32_t action, value; };
struct FeedUiReply { uint32_t bytes; }; // Bounded UTF-8 JSON follows (at most 1 MiB).
struct FeedSettingsPatch { uint32_t bytes; }; // 'S': live-only JSON, <= 8192 bytes; uint32 acceptance reply.

#pragma pack(pop)

static_assert(sizeof(FeedBuild)==116);
static_assert(sizeof(FeedFrameMsg)==24);
static_assert(sizeof(FeedBuildAck)==24);
