#include <ehs.hpp>

#include <pokerTypes.hpp>

#include <chrono>
#include <cstring>
#include <iostream>

#include <sys/mman.h>
#include <errno.h>

int main()
{
    using namespace PokerTypes;

    constexpr size_t HugePageSize = 2 * 1024 * 1024;
    constexpr uint64_t _7C2 = 21;
    constexpr uint64_t nCanonicalHands = 2'428'287'420;
    constexpr uint64_t numElements = nCanonicalHands*_7C2;
    constexpr size_t fileSize = numElements * sizeof(uint16_t);

    FILE* fd = fopen("ehs.dat", "w+");
    if (fseek(fd, fileSize-1, SEEK_SET) != 0)
    {
        std::cerr << "fseek failed." << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }

    if (fwrite("\0", 1, 1, fd) < 1)
    {
        std::cerr << "fwrite failed." << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }

    uint16_t* riverEHS = (uint16_t*)mmap(NULL, fileSize, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fileno(fd), 0);
    fclose(fd);


    // uint8_t cards[7] = {1,5,9,13,17,21,25};
    // int u0 = HR[53+cards[0]];
    // int u1 = HR[u0+cards[1]];
    // int u2 = HR[u1+cards[2]];
    // int u3 = HR[u2+cards[3]];
    // int u4 = HR[u3+cards[4]]; // NOTE (keb): HR[u4] is the 5-card hand rank
    // int u5 = HR[u4+cards[5]]; // NOTE (keb): HR[u5] is the 6-card hand rank
    // int u6 = HR[u5+cards[6]]; // NOTE (keb): u6 is the 7-card hand rank

    if (riverEHS == MAP_FAILED) {
        std::cerr << "Memory allocation failed!" << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // EHS::generateEHSRiver(riverEHS);
    EHS::generateEHSRiverCanonical(riverEHS);
    // for (int i = 0; i < 100'000; ++i)
    // {
    //     deck.shuffle(); 
    //     std::array<PokerTypes::Card, 2> myCards {deck[0], deck[1]};
    //     std::array<PokerTypes::Card, 5> boardCards {deck[2], deck[3], deck[4], deck[5], deck[6]};
    //     EHS::getEHS(myCards, boardCards);
    // }
    // hand_indexer_t river_indexer;
    // assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));
    // uint8_t cards[7];
    // hand_unindex(&river_indexer, 3, 0, cards);
    // std::bitset<52> deck;
    // for (int i = 0; i < 7; ++i)
    // {
    //     deck.flip(cards[i]);
    // }
    // std::cout << "Manual EHS: " << EHS::getEHS({Card(cards[1]), Card(cards[4])}, {Card(cards[5]), Card(cards[3]),Card(cards[2]), Card(cards[0]),Card(cards[6])}, deck) << std::endl;
    
    // hand_unindex(&river_indexer, 3, 3, cards);
    // deck.reset();
    // for (int i = 0; i < 7; ++i)
    // {
    //     deck.flip(cards[i]);
    // }
    // std::cout << "Manual EHS: " << EHS::getEHS({Card(cards[0]), Card(cards[1])}, {Card(cards[2]), Card(cards[3]),Card(cards[4]), Card(cards[5]),Card(cards[6])}, deck) << std::endl;
    
    // hand_unindex(&river_indexer, 3, 6, cards);
    // deck.reset();
    // for (int i = 0; i < 7; ++i)
    // {
    //     deck.flip(cards[i]);
    // }
    // std::cout << "Manual EHS: " << EHS::getEHS({Card(cards[0]), Card(cards[1])}, {Card(cards[2]), Card(cards[3]),Card(cards[4]), Card(cards[5]),Card(cards[6])}, deck) << std::endl;
    
    // hand_unindex(&river_indexer, 3, 9, cards);
    // deck.reset();
    // for (int i = 0; i < 7; ++i)
    // {
    //     deck.flip(cards[i]);
    // }
    // std::cout << "Manual EHS: " << EHS::getEHS({Card(cards[0]), Card(cards[1])}, {Card(cards[2]), Card(cards[3]),Card(cards[4]), Card(cards[5]),Card(cards[6])}, deck) << std::endl;

    // std::cout << "MMAP EHS: " << riverEHS[10] << std::endl;
    // std::cout << "MMAP EHS: " << riverEHS[63] << std::endl;
    // std::cout << "MMAP EHS: " << riverEHS[126] << std::endl;
    // std::cout << "MMAP EHS: " << riverEHS[189] << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end-start).count() << std::endl;
    
    // Don't forget to free the memory when done
    munmap(riverEHS, numElements * sizeof(uint16_t));
}