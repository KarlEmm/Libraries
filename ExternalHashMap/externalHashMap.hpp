#pragma once

#include <memory.hpp>

#include <iostream>
#include <math.h>
#include <optional>
#include <utility>
#include <vector>

namespace ExtHM
{

template <typename K, typename V>
requires std::is_copy_constructible_v<V>
class ExternalHashMap
{
    using value_type = std::pair<K, V>;

public:
    ExternalHashMap(const char* filename, uint64_t nItems) : m_size(findMiniumMapSize(nItems))
    {
        m_data = Memory::getMmap<value_type>(filename, m_size, false);
        m_occupied = std::vector<uint8_t>((m_size / 8 + 1), 0);
        assert(m_data && "Couldn't get a backing file for the HashMap");
    }

    std::pair<std::reference_wrapper<V>, bool> insert(const K& k, const V& v)
    {
        uint64_t index = k % m_size;
        bool collided = false;
        for (uint64_t i = 0; i < m_size; ++i)
        {
            if (!isOccupied(index))
            {
                setOccupied(index);
                m_data[index] = {k, v};
                return {m_data[index].second, true};
            }

            if (m_data[index].first == k)
            {
                return {m_data[index].second, false};
            }

            if (!collided)
            {
                collided = true;
                ++m_collisions;
            }

            // Linear Probing
            index = (index + 1) % m_size;
        }
        assert(false && "Failed to insert element in ExternalHashMap");
        std::unreachable();
    }
    
    std::optional<std::reference_wrapper<V>> get(const K& k)
    {
        uint64_t index = k % m_size;
        for (uint64_t i = 0; i < m_size; ++i)
        {
            if (m_data[index].first == k)
            {
                return m_data[index].second;
            }

            // Linear Probing
            index = (index + 1) % m_size;
        }

        return std::nullopt;
    }

    uint64_t size()
    {
        return m_size;
    }

    struct Iterator
    {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = ExternalHashMap::value_type;
        using pointer = value_type*;
        using reference = value_type&;

        Iterator(pointer ptr, ExternalHashMap<K, V>* container) : 
            m_ptr{ptr}, m_container{container} {}

        reference operator*() const { return *m_ptr; }
        pointer operator->() { return m_ptr; }


        Iterator& operator++() 
        { 
            ++m_ptr; 
            std::ptrdiff_t index = *this - m_container->begin();
            while (!m_container->isOccupied(index) && *this != m_container->end())
            {
                ++m_ptr; 
                index = *this - m_container->begin();
            }
            return *this; 
        }
        Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; } 

        friend bool operator==(const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
        friend bool operator!=(const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };
        friend std::ptrdiff_t operator-(const Iterator& a, const Iterator& b) { return a.m_ptr - b.m_ptr; };

    private:
        pointer m_ptr;
        ExternalHashMap* m_container;
    };

    Iterator begin() { return Iterator(m_data.get(), this); }
    Iterator end() { return Iterator(m_data.get() + m_size, this); }

private:
    bool isPrime(uint64_t n)
    {
        uint64_t nSqrt = sqrt(n) + 1;
        for (uint64_t i = 3; i <= nSqrt; ++i)
        {
            if (n % i == 0)
            {
                return false;
            }
        }
        return true;
    }

    uint64_t findMiniumMapSize(uint64_t nItems)
    {
        uint64_t candidate = nItems;
        // Only test odd numbers 
        if (candidate % 2 == 0)
        {
            ++candidate;
        }

        // https://en.wikipedia.org/wiki/Bertrand%27s_postulate
        while (true)
        {
            if (isPrime(candidate))
            {
                return candidate;
            }
            candidate += 2;
        }

        std::unreachable();
    }

    void setOccupied(uint64_t index)
    {
        m_occupied[(index / 8)] |= (1 << index % 8);
    }

    bool isOccupied(uint64_t index)
    {
        return m_occupied[(index / 8)] & (1 << index % 8);
    }

    std::vector<uint8_t> m_occupied; 
    uint64_t m_size {0};
    std::unique_ptr<value_type[], Memory::MmapDeleter<value_type>> m_data;

    // TODO (keb): remove, only for analytics
    uint64_t m_collisions {0};

    friend Iterator;
};

}; // namespace ExtHM