#include "memory.hpp"
#include <iostream>

int main()
{
    auto mem = Memory::getMmap<uint16_t>("memorytest.dat", 200, true);

    std::cout << mem[150] << std::endl; 
}
