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

}