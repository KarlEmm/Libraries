#include <externalHashMap.hpp>

#include <iostream>

int main()
{
    ExtHM::ExternalHashMap<int, int> m ("testmap.dat", 100);

    for (uint64_t i = 0; i < 50; ++i)
    {
        auto [v, inserted] = m.insert(i * 3, i);
        v += 1;
    }

    for (auto e : m)
    {
        std::cout << e.first << ", " << e.second << std::endl;
    }
}