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

    hand_indexer_t river_indexer;
    assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));
    uint64_t nCanonicalHands = hand_indexer_size(&river_indexer, 1);

    nCanonicalHands = hand_indexer_size(&river_indexer, 0);
    uint8_t cards[2];
    for (int i = 0; i < nCanonicalHands; ++i)
    {
        hand_unindex(&river_indexer, 0, i, cards);
        std::cout << i << ":  ";
        for (int j = 0; j < 2; ++j)
        {
            std::cout << (char)RANK_TO_CHAR[cards[j] >> 2] << (char)SUIT_TO_CHAR[cards[j] & 3] << " ";
        }
        std::cout << std::endl;
    }

}