#include <ehs.hpp>

extern "C"
{
    #include <hand_index.h>
}

#include <pokerTypes.hpp>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>

#include <sys/mman.h>
#include <errno.h>

namespace TMP 
{
    const char* riverEHSFilename = "ehs.dat";
    const char* flopHistogramsFilename = "flopHistograms.dat";
    const char* turnHistogramsFilename = "turnHistograms.dat";
    const char* flopCentroidsFilename = "flopCentroids.dat";
    const char* turnCentroidsFilename = "turnCentroids.dat"; 

    constexpr int nBuckets = 200;
};

int main()
{
    using namespace PokerTypes;

    // EHS::createRoundsAbstractions();

    // EHS::generateFlopHistograms();
    // EHS::finalizeHistograms(1, TMP::flopHistogramsFilename);
    // EHS::writeRoundCentroids(TMP::flopCentroidsFilename, TMP::flopHistogramsFilename, TMP::nBuckets, 1);
    
    //  EHS::finalizeHistograms(2, TMP::turnHistogramsFilename);

    hand_indexer_t river_indexer;
    assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));
    uint64_t nCanonicalHands = hand_indexer_size(&river_indexer, 1);

    nCanonicalHands = TMP::nBuckets;
    auto mp = Memory::getMmap<Histogram<50>>(TMP::turnCentroidsFilename, nCanonicalHands);
    for (int i = 0; i < nCanonicalHands; ++i)
    {
        const auto& h = mp[i];
        for (int j = 0; j < h.size(); ++j)
        {
            std::cout << h[j] << ", ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

}