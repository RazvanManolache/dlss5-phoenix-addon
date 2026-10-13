// Creates native CUDA kernel handles and owns their driver lifetime.
// Bindings and launch geometry come from the validated model/kernel contract, not from filename guesses.

#pragma once
#include "device_recovery.h"
#include "../support/kernel_resolution.h"
#include "../../../third_party/nvapi/nvapi.h"
#include <bcrypt.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#pragma comment(lib, "bcrypt.lib")

namespace phoenix_kernel {
struct Spec {
    std::string name, path, sha, entry;
    unsigned bytes, x, y, z;
    // Both entries live in one module and share the same arguments and geometry.
    // Older bundles omit these fields and keep their original entry point.
    std::string large_input_entry;
    uint64_t large_input_min_pixels = 0;
};
struct Api {
    decltype(&NvAPI_D3D12_CreateCuModule) create_module = nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) create_function = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroy_function = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroy_module = nullptr;
    void validate() const {
        if (!create_module || !create_function || !destroy_function || !destroy_module)
            throw std::invalid_argument("Lifecycle37 requires original NVAPI entry points");
    }
};
struct Counters {
    std::atomic<uint64_t> modules_created{0}, functions_created{0}, module_destroy_attempts{0},
        function_destroy_attempts{0};
    std::atomic<uint64_t> modules_destroyed{0}, functions_destroyed{0}, destroy_failures{0};
};
struct Binary {
    Spec spec;
    std::vector<unsigned char> bytes;
};
// Verify every required module against its size, entry point and SHA256 pin
// before creating driver handles. The catalog must contain all 41 modules.
inline std::vector<Binary> load_catalog(const std::vector<Spec>& specs) {
    if (specs.size() != 41)
        throw std::invalid_argument("Lifecycle37 requires the complete 41-kernel catalog");
    std::vector<Binary> result;
    result.reserve(specs.size());
    std::unordered_map<std::string, bool> names;
    for (const auto& specification : specs) {
        if (!names.emplace(specification.name, true).second || specification.sha.size() != 64 ||
            specification.entry.empty() || !specification.bytes || !specification.x ||
            !specification.y || !specification.z ||
            !valid_resolution_variant(specification.entry, specification.large_input_entry,
                                      specification.large_input_min_pixels))
            throw std::invalid_argument("Lifecycle37 invalid or duplicate kernel contract");
        std::ifstream file(specification.path, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("Lifecycle37 missing kernel " + specification.path);
        auto size = file.tellg();
        if (size <= 0 || size > 16 * 1024 * 1024)
            throw std::runtime_error("Lifecycle37 kernel size rejected");
        Binary binary{specification, std::vector<unsigned char>(static_cast<size_t>(size))};
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(binary.bytes.data()), binary.bytes.size()))
            throw std::runtime_error("Lifecycle37 short kernel read");
        unsigned char hash[32];
        char hex[65]{};
        if (BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0, binary.bytes.data(),
                       ULONG(binary.bytes.size()), hash, 32) < 0)
            throw std::runtime_error("Lifecycle37 SHA256 failed");
        for (unsigned iteration_index = 0; iteration_index < 32; ++iteration_index)
            sprintf_s(hex + iteration_index * 2, 3, "%02x", unsigned(hash[iteration_index]));
        if (specification.sha != hex)
            throw std::runtime_error("Lifecycle37 kernel pin mismatch " + specification.name);
        result.push_back(std::move(binary));
    }
    return result;
}
struct Module {
    ID3D12Device* device;
    Api api;
    std::shared_ptr<Counters> counters;
    NVDX_ObjectHandle module = nullptr, function = nullptr;
    Module(ID3D12Device* device_handle, Api kernel_api,
           std::shared_ptr<Counters> lifecycle_counters)
        : device(device_handle), api(kernel_api), counters(std::move(lifecycle_counters)) {}
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    ~Module() {
        if (function) {
            ++counters->function_destroy_attempts;
            if (api.destroy_function(device, function) == NVAPI_OK)
                ++counters->functions_destroyed;
            else
                ++counters->destroy_failures;
        }
        if (module) {
            ++counters->module_destroy_attempts;
            if (api.destroy_module(device, module) == NVAPI_OK)
                ++counters->modules_destroyed;
            else
                ++counters->destroy_failures;
        }
    }
};
struct Payload final : phoenix_device::Payload {
    std::unordered_map<std::string, std::unique_ptr<Module>> modules;
    std::unordered_map<uint64_t, std::string> native_names;
    std::mutex names_mutex;
};

// Owns actual module/function handles, not the legacy process-lived Candidate
// cache. Frame leases are the only permitted asynchronous users of these handles.
// All native NGX input/guide/history allocations remain engine-owned.
class Network {
    Api api_;
    std::vector<Binary> catalog_;
    std::unordered_map<std::string, const Binary*> by_name_;
    phoenix_device::BorrowedBinding binding_;
    std::shared_ptr<Counters> counters_ = std::make_shared<Counters>();

  public:
    using Lease = phoenix_device::BorrowedBinding::Lease;
    Network(Api api, std::vector<Spec> specs) : api_(api), catalog_(load_catalog(specs)) {
        api_.validate();
        for (const auto& binary : catalog_)
            by_name_.emplace(binary.spec.name, &binary);
    }
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    std::shared_ptr<Counters> counters() const {
        return counters_;
    }
    const std::vector<Binary>& catalog() const {
        return catalog_;
    }
    Lease rebind(ID3D12Device* device) {
        return binding_.rebind(
            device,
            [&](ID3D12Device* rebound_device,
                uint64_t) -> std::unique_ptr<phoenix_device::Payload> {
                auto payload = std::make_unique<Payload>();
                for (const auto& binary : catalog_) {
                    auto value = std::make_unique<Module>(rebound_device, api_, counters_);
                    auto status = api_.create_module(rebound_device, binary.bytes.data(),
                                                     NvU32(binary.bytes.size()), &value->module);
                    if (status != NVAPI_OK)
                        throw std::runtime_error("Lifecycle37 CreateCuModule rejected " +
                                                 binary.spec.name);
                    ++counters_->modules_created;
                    status = api_.create_function(rebound_device, value->module,
                                                  binary.spec.entry.c_str(), &value->function);
                    if (status != NVAPI_OK)
                        throw std::runtime_error("Lifecycle37 CreateCuFunction rejected " +
                                                 binary.spec.name);
                    ++counters_->functions_created;
                    payload->modules.emplace(binary.spec.name, std::move(value));
                }
                if (payload->modules.size() != 41)
                    throw std::runtime_error("Lifecycle37 incomplete device rebuild");
                return payload;
            });
    }
    Lease acquire(ID3D12Device* device) {
        return binding_.acquire(device);
    }
    bool allowed(const Lease& lease) const {
        return binding_.dispatch_allowed(lease);
    }
    void invalidate(HRESULT reason) {
        binding_.invalidate(reason);
    }
    bool observe(ID3D12Device* device, HRESULT result, uint64_t completed = 0) {
        return binding_.observe(device, result, completed);
    }
    bool register_native(const Lease& lease, const std::string& name, NVDX_ObjectHandle function) {
        if (!allowed(lease) || !function || !by_name_.count(name))
            return false;
        auto& payload = *static_cast<Payload*>(lease->resources.get());
        std::lock_guard<std::mutex> lock(payload.names_mutex);
        payload.native_names[uint64_t(function)] = name;
        return true;
    }
    void erase_native(const Lease& lease, NVDX_ObjectHandle function) {
        if (!lease)
            return;
        auto& payload = *static_cast<Payload*>(lease->resources.get());
        std::lock_guard<std::mutex> lock(payload.names_mutex);
        payload.native_names.erase(uint64_t(function));
    }
    NVDX_ObjectHandle resolve(const Lease& lease,
                              const NVAPI_CU_KERNEL_LAUNCH_PARAMS& launch) const {
        if (!allowed(lease))
            throw std::runtime_error("Lifecycle37 stale dispatch epoch");
        auto& payload = *static_cast<Payload*>(lease->resources.get());
        std::string name;
        {
            std::lock_guard<std::mutex> lock(payload.names_mutex);
            auto lookup_iterator = payload.native_names.find(uint64_t(launch.hFunction));
            if (lookup_iterator == payload.native_names.end())
                throw std::runtime_error("Lifecycle37 unknown native function");
            name = lookup_iterator->second;
        }
        const auto& spec = by_name_.at(name)->spec;
        if (!launch.pParams || launch.paramSize != spec.bytes || launch.blockDim.x != spec.x ||
            launch.blockDim.y != spec.y || launch.blockDim.z != spec.z || launch.dynSharedMemBytes)
            throw std::runtime_error("Lifecycle37 launch contract mismatch");
        return payload.modules.at(name)->function;
    }
};
}
