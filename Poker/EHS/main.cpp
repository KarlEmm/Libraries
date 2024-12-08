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
    const char* riverOCHSFilename = "riverOchs.dat";
    const char* flopHistogramsFilename = "flopHistograms.dat";
    const char* turnHistogramsFilename = "turnHistograms.dat";
    const char* flopCentroidsFilename = "flopCentroids.dat";
    const char* turnCentroidsFilename = "turnCentroids.dat"; 
    const char* riverCentroidsFilename = "riverCentroids.dat"; 

    constexpr int nBuckets = 200;
    constexpr int nEHSHistogramsBins = 50;
    constexpr int nOCHSHistogramsBins = 8;
};

int main()
{
    using namespace PokerTypes;

    hand_indexer_t river_indexer;
    assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));
    uint64_t nCanonicalHands = hand_indexer_size(&river_indexer, 3);

    // EHS::generateRiverOCHS();
    EHS::writeRoundCentroids<TMP::nOCHSHistogramsBins, KMeans::L2Distance<Histogram<TMP::nOCHSHistogramsBins>>>(TMP::riverCentroidsFilename,
        TMP::riverOCHSFilename, 
        TMP::nBuckets, 3);

}