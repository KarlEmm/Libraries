#pragma once

#include <random.hpp>

#include <array>
#include <cstdint>
#include <string>

namespace PokerTypes 
{

struct Histogram
{
	Histogram& operator+=(const Histogram& other)
	{
        auto thisItr = m_data.begin();
        auto otherItr = other.m_data.begin();

        for (; thisItr != m_data.end() && otherItr != other.m_data.end(); ++thisItr, ++otherItr)
        {
            *thisItr += *otherItr;
        }
        return *this;
	}
    
	Histogram operator/(double divisor)
    {
        Histogram result;
		for (int i = 0; i < m_data.size(); ++i)
		{
            result.m_data[i] = (m_data[i] / divisor);
        }
        return result;
    }

	size_t size() const { return m_data.size(); }
	uint8_t& operator[](size_t index) { return m_data[index]; }
	const uint8_t& operator[](size_t index) const { return m_data[index]; }

	std::array<uint8_t, 20> m_data;
};

namespace Constants
{
	constexpr int nPlayers{ 2 };
	constexpr int nFlopCards{ 3 };
	constexpr int nTurnCards{ 1 };
	constexpr int nRiverCards{ 1 };
	constexpr int nCommunityCards{ nFlopCards + nTurnCards + nRiverCards };
	constexpr int nPrivateCards{ 2 };
    constexpr int nTotalPlayerCards = nPrivateCards * nPlayers;
	constexpr int nTotalCards = nTotalPlayerCards + nCommunityCards;
	constexpr int communityCardIndex = nPrivateCards * nPlayers;

	constexpr int nRanks{ 13 };
	constexpr int nSuits{ 4 };
	constexpr int nDeckCards{ nRanks * nSuits };
	const char RANK_TO_CHAR[] = "23456789TJQKA";
	const char SUIT_TO_CHAR[] = "shdc";
};

enum class BettingRound : uint8_t
{
	Preflop,
	Flop,
	Turn,
	River,
	RoundCount
};

struct Card
{
	Card() : rank{ 0 }, suit{ 0 } {};
	explicit Card(uint8_t card) : rank{ static_cast<uint8_t>(card >> 2) }, suit{ static_cast<uint8_t>(card & 0b11) } {};
	uint8_t rank : 6;
	uint8_t suit : 2;

    std::string to_string() 
	{
		std::string result = std::string{static_cast<char>(Constants::RANK_TO_CHAR[rank]), static_cast<char>(Constants::SUIT_TO_CHAR[suit])}; 
		return result;
	}

	operator int() { return (rank << 2) | (suit); }
	operator int() const { return (rank << 2) | (suit); }
};

struct Deck
{
    Deck()
    {
        init();
    }

	void init()
	{
		for (int i = Constants::nDeckCards - 1; i >= 0; --i)
		{
			deck[i] = Card{ static_cast<uint8_t>(i)};
		}
	}

	void shuffle()
	{
		for (int i = Constants::nDeckCards - 1; i >= 0; --i)
		{
			int rand = Random::get(0, i);
			std::swap(deck[i], deck[rand]);
		}
	}

    void randomizePlayerCards(int playerIndex)
    {
        for (int playerCardIndex = playerIndex * 2; playerCardIndex <= playerIndex * 2 + 1; ++playerCardIndex)
        {
			int rand = Random::get(firstUnusedCardIndex(), Constants::nDeckCards);
			std::swap(deck[playerCardIndex], deck[rand]);
        }
    }
    
    int firstUnusedCardIndex()
    {
        return Constants::nTotalCards;
    }

    void setPlayerCards(int playerIndex, const std::array<Card, Constants::nPrivateCards>& cards)
    {
        std::swap(deck[playerIndex * 2], deck[static_cast<const int>(cards[0])]);
        std::swap(deck[playerIndex * 2 + 1], deck[static_cast<const int>(cards[1])]);
    }
    
    void setCommunityCards(const std::array<Card, Constants::nCommunityCards>& cards)
    {
        for (int i = 0; i < Constants::nCommunityCards; ++i)
        {
            std::swap(deck[Constants::nTotalPlayerCards + i], deck[static_cast<const int>(cards[i])]);
        }
    }
	
	std::array<Card, Constants::nPrivateCards> getPreflopHand(int playerIndex) const { return { deck[playerIndex * 2], deck[playerIndex * 2 + 1] }; } 
	std::array<Card, Constants::nFlopCards> getFlopCards() const { return { deck[Constants::communityCardIndex], deck[Constants::communityCardIndex+1], deck[Constants::communityCardIndex+2] }; } 
	std::array<Card, Constants::nTurnCards> getTurnCards() const { return { deck[Constants::communityCardIndex + 3] }; } 
	std::array<Card, Constants::nRiverCards> getRiverCards() const { return { deck[Constants::communityCardIndex + 4] }; } 
	std::array<Card, Constants::nPrivateCards + Constants::nFlopCards> getFlopHand(int playerIndex) const
	{
		const auto& preflopCards = getPreflopHand(playerIndex);
		const auto& flopCards = getFlopCards();
		
		std::array<Card, Constants::nPrivateCards + Constants::nFlopCards> flopHand;
		std::copy(preflopCards.begin(), preflopCards.end(), flopHand.begin());
		std::copy(flopCards.begin(), flopCards.end(), flopHand.begin() + Constants::nPrivateCards);

		return flopHand;
	}
	std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards> getTurnHand(int playerIndex) const
	{
		const auto& preflopCards = getPreflopHand(playerIndex);
		const auto& flopCards = getFlopCards();
		const auto& turnCards = getTurnCards();
		
		std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards> turnHand;
		std::copy(preflopCards.begin(), preflopCards.end(), turnHand.begin());
		std::copy(flopCards.begin(), flopCards.end(), turnHand.begin() + Constants::nPrivateCards);
		std::copy(turnCards.begin(), turnCards.end(), turnHand.begin() + Constants::nPrivateCards + Constants::nFlopCards);

		return turnHand;
	}
	std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards + Constants::nRiverCards> getRiverHand(int playerIndex) const
	{
		const auto& preflopCards = getPreflopHand(playerIndex);
		const auto& flopCards = getFlopCards();
		const auto& turnCards = getTurnCards();
		const auto& riverCards = getRiverCards();
		
		std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards + Constants::nRiverCards> riverHand;
		std::copy(preflopCards.begin(), preflopCards.end(), riverHand.begin());
		std::copy(flopCards.begin(), flopCards.end(), riverHand.begin() + Constants::nPrivateCards);
		std::copy(turnCards.begin(), turnCards.end(), riverHand.begin() + Constants::nPrivateCards + Constants::nFlopCards);
		std::copy(riverCards.begin(), riverCards.end(), riverHand.begin() + Constants::nPrivateCards + Constants::nFlopCards + Constants::nRiverCards);

		return riverHand;
	}

	Card operator[](int index) { return deck[index]; }

	std::array<Card, Constants::nDeckCards> deck{ };

    // Bit Representation
    uint64_t cards {0};

    bool isCardUsed(Card c) { return cards & (1 << static_cast<int>(c)); }

    // ==================
};

};