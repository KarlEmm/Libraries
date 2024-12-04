#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <sys/mman.h>

namespace Memory
{

template <typename T>
struct MmapDeleter
{
    explicit MmapDeleter(size_t size) : m_size{size} {};
    MmapDeleter(const MmapDeleter& other) = default;
    MmapDeleter(MmapDeleter&& other) = default;
    MmapDeleter& operator=(const MmapDeleter&) = default;
    MmapDeleter& operator=(MmapDeleter&&) = default;

    void operator()(T* p) const
    {
        munmap(p, m_size);
    };

    size_t m_size {0};
};

template <typename T, typename TMmapDeleter = MmapDeleter<T>>
std::unique_ptr<T[], TMmapDeleter> getMmap(const char* filename, size_t nElements, bool readonly = true)
{
    size_t mappedSize = nElements * sizeof(T);
    FILE* fd = fopen(filename, readonly ? "r" : "w+x");

    if (!fd)
    {
        return {nullptr, TMmapDeleter{mappedSize}};
    }

    if (!readonly)
    {
        // NOTE (keb): Create a file if it doesn't already exist and reserve memory.
        if (fseek(fd, mappedSize-1, SEEK_SET) != 0)
        {
            return {nullptr, TMmapDeleter{mappedSize}};
        }

        if (fwrite("\0", 1, 1, fd) < 1)
        {
            return {nullptr, TMmapDeleter{mappedSize}};
        }
    }

    T* mmapPtr = static_cast<T*>(mmap(NULL, mappedSize, 
        readonly ? PROT_READ : PROT_READ | PROT_WRITE,
        MAP_SHARED, fileno(fd), 0));

    fclose(fd);

    if (mmapPtr == MAP_FAILED) 
    {
        return {nullptr, TMmapDeleter{mappedSize}};
    }

    return {mmapPtr, TMmapDeleter{mappedSize}};
}
}; // namespace Memory
