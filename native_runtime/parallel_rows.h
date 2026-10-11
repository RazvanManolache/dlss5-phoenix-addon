#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace phoenix_cpp {
// One persistent pool per frame state. Rows never share output pixels, so
// dispatching them in a different order does not change reduction arithmetic.
class RowExecutor {
    std::mutex dispatch_mutex_;
    std::condition_variable rows_ready_, rows_finished_;
    std::vector<std::thread> workers_;
    std::function<void(unsigned)> row_function_;
    std::atomic<unsigned> next_row_{0};
    unsigned row_count_ = 0, unfinished_workers_ = 0;
    uint64_t dispatch_generation_ = 0;
    bool stopping_ = false;
    std::exception_ptr failure_;

    void worker() {
        uint64_t observed_generation = 0;
        std::unique_lock lock(dispatch_mutex_);
        for (;;) {
            rows_ready_.wait(
                lock, [&] { return stopping_ || observed_generation != dispatch_generation_; });
            if (stopping_)
                return;
            observed_generation = dispatch_generation_;
            lock.unlock();
            try {
                for (unsigned first; (first = next_row_.fetch_add(4)) < row_count_;)
                    for (unsigned row = first; row < std::min(first + 4, row_count_); ++row)
                        row_function_(row);
            } catch (...) {
                std::lock_guard guard(dispatch_mutex_);
                if (!failure_)
                    failure_ = std::current_exception();
            }
            lock.lock();
            if (--unfinished_workers_ == 0)
                rows_finished_.notify_one();
        }
    }

  public:
    explicit RowExecutor(
        unsigned count = std::min(16u, std::max(1u, std::thread::hardware_concurrency() / 2))) {
        count = std::max(1u, count);
        try {
            for (unsigned iteration_index = 0; iteration_index < count; ++iteration_index)
                workers_.emplace_back([this] { worker(); });
        } catch (...) {
            {
                std::lock_guard guard(dispatch_mutex_);
                stopping_ = true;
            }
            rows_ready_.notify_all();
            for (auto& thread : workers_)
                thread.join();
            throw;
        }
    }
    ~RowExecutor() {
        {
            std::lock_guard guard(dispatch_mutex_);
            stopping_ = true;
        }
        rows_ready_.notify_all();
        for (auto& thread : workers_)
            thread.join();
    }
    RowExecutor(const RowExecutor&) = delete;
    RowExecutor& operator=(const RowExecutor&) = delete;

    // Run small jobs inline. Larger jobs reuse persistent workers; the barrier
    // publishes every output row and propagates the first worker exception.
    template <class Function> void run(unsigned rows, Function function) {
        if (rows < 64) {
            for (unsigned row = 0; row < rows; ++row)
                function(row);
            return;
        }
        std::unique_lock lock(dispatch_mutex_);
        row_function_ = std::move(function);
        row_count_ = rows;
        next_row_ = 0;
        unfinished_workers_ = unsigned(workers_.size());
        failure_ = nullptr;
        ++dispatch_generation_;
        rows_ready_.notify_all();
        rows_finished_.wait(lock, [&] { return unfinished_workers_ == 0; });
        row_function_ = {};
        if (failure_)
            std::rethrow_exception(failure_);
    }
};
}
