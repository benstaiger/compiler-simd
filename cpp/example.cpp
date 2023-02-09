#include <chrono>
#include <iostream>

#include "Eigen/Dense"

using namespace Eigen;

int main() {
    VectorXf x(1024*1024);
    VectorXf y(1024*1024);

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    const size_t iterations = 10'000;
    [[maybe_unused]] float res = 0;
    auto t1 = high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        res = x.dot(y);  // does not seem to produce a fused mul-add on aarch64
    }
    auto t2 = high_resolution_clock::now();

    duration<double, std::micro> ms_double = t2 - t1;

    // certainly some overhead included, but should be minimal.
    std::cout << ms_double.count()/iterations << "us\n";
    std::cout << res << std::endl;
    return 0;
}