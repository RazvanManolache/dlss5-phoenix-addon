#pragma once
#include "capture_queue.h"
#include <array>
#include <d3d9.h>
#include <d3d11.h>
#include <stdexcept>
#include <wrl/client.h>

namespace phoenix_compat {
// A marker for each pending snapshot, one staging copy and one upload.
// Serial order identifies each copy without polling unrelated GPU work.
template<class Query>
class TransferEventQueries {
    std::array<Microsoft::WRL::ComPtr<Query>, CaptureQueue::max_slots + 2> queries_;
    uint64_t issued_ = 0;
    uint64_t completed_ = 0;

public:
    explicit TransferEventQueries(auto&& create) {
        for (auto& query : queries_)
            if (FAILED(create(query.GetAddressOf())))
                throw std::runtime_error("Cannot allocate graphics transfer queries");
    }

    uint64_t completed(auto&& poll) {
        while (completed_ < issued_) {
            const auto result = poll(queries_[completed_ % queries_.size()].Get());
            if (result == S_FALSE)
                break;
            if (result != S_OK)
                throw std::runtime_error("Cannot poll graphics transfer completion");
            ++completed_;
        }
        return completed_;
    }

    bool signal(uint64_t value, auto&& issue) {
        if (value != issued_ + 1 || issued_ - completed_ >= queries_.size())
            return false;
        if (FAILED(issue(queries_[issued_ % queries_.size()].Get())))
            return false;
        issued_ = value;
        return true;
    }
};

class D3D9TransferFence {
    TransferEventQueries<IDirect3DQuery9> queries_;

public:
    explicit D3D9TransferFence(IDirect3DDevice9* device)
        : queries_([device](auto** query) { return device->CreateQuery(D3DQUERYTYPE_EVENT, query); }) {}
    uint64_t completed() {
        // Present submits the queries. Polling must never force a driver flush.
        return queries_.completed([](auto* query) { return query->GetData(nullptr, 0, 0); });
    }
    bool signal(uint64_t value) {
        return queries_.signal(value, [](auto* query) { return query->Issue(D3DISSUE_END); });
    }
};

class D3D11TransferFence {
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
    TransferEventQueries<ID3D11Query> queries_;

public:
    D3D11TransferFence(ID3D11Device* device, ID3D11DeviceContext* context)
        : context_(context), queries_([device](auto** query) {
            const D3D11_QUERY_DESC desc{D3D11_QUERY_EVENT, 0};
            return device->CreateQuery(&desc, query);
        }) {}
    uint64_t completed() {
        return queries_.completed([this](auto* query) {
            return context_->GetData(query, nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
        });
    }
    bool signal(uint64_t value) {
        return queries_.signal(value, [this](auto* query) {
            context_->End(query);
            return S_OK;
        });
    }
};
} // namespace phoenix_compat
