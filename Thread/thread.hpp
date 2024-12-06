#include <thread>
#include <vector>

namespace Thread
{

template <typename TCallback>
void startThreadedLoop(TCallback& callback, size_t nElements, int nThreads = 1)
{
    size_t nElementsPerThread = nElements / nThreads;
    size_t nElementsLeftover = nElements % nThreads;
    std::vector<std::thread> threads;
    for (int i = 0; i < nThreads; ++i)
    {
        if (i == nThreads-1)
        {
            threads.emplace_back(callback, i*nElementsPerThread, i*nElementsPerThread+nElementsPerThread+nElementsLeftover, i);
            break;
        }
        threads.emplace_back(callback, i*nElementsPerThread, i*nElementsPerThread+nElementsPerThread, i);
    }
    for_each(threads.begin(), threads.end(), [](auto& thread){ thread.join(); });
}

}; // namespace Thread