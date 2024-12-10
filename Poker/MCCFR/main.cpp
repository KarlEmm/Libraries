#include <kmeans.hpp>
#include <memory.hpp>
#include <pokerTypes.hpp>
#include <random.hpp>
#include <time.hpp>

#include <phevaluator/phevaluator.h>
extern "C"
{
    #include <hand_index.h>
}

#include <array>
#include <cassert>
#include <chrono>
#include <iostream> // For debugging
#include <numeric>
#include <unordered_map>
#include <unordered_set>


namespace MCCFR
{
using namespace PokerTypes;

std::array<hand_indexer_t, (uint8_t)BettingRound::RoundCount> indexers;

std::span<Histogram<AbstractionsContext::nEHSHistogramsBins>> flopHistograms;
std::span<Histogram<AbstractionsContext::nEHSHistogramsBins>> turnHistograms;
std::span<Histogram<AbstractionsContext::nOCHSHistogramsBins>> riverHistograms;

std::span<Histogram<AbstractionsContext::nEHSHistogramsBins>> flopCentroids;
std::span<Histogram<AbstractionsContext::nEHSHistogramsBins>> turnCentroids;
std::span<Histogram<AbstractionsContext::nOCHSHistogramsBins>> riverCentroids;

inline std::chrono::time_point<std::chrono::steady_clock> startTime;

template <typename T>
concept SmallEnum = requires ()
{
	{ std::is_enum_v<T> };
	{ std::is_convertible_v<T, uint8_t> };
};

template <typename T>
requires SmallEnum<T> || (std::is_integral_v<T> && sizeof(T) <= 8)
struct Flags8Bits
{
	uint8_t flags { 0b0000'0000 };

	void set(T index) { flags |= 1 << index; }
	void unset(T index) { flags &= ~(1 << index); }
	bool test(T index) { return flags & (1 << index); }
	bool nSet() { return __builtin_popcount(flags); }
	bool one() { return nSet() == 1; }
};

std::string toString(Action action)
{
	if (action == Fold)
		return "F";
	if (action == Call)
		return "C";
	if (action == Check)
		return "CH";
	if (action == BetQuarter)
		return "BQ";
	if (action == BetHalf)
		return "BH";
	if (action == BetPot)
		return "BP";
	if (action == BetOver)
		return "BO";
	return "?";
}

using Actions = Flags8Bits<Action>;

namespace Math
{
	bool isAlmostEqual(double a, double b, double epsilon = 0.00001)
	{
		return std::abs(a - b) < epsilon;
	}
};

namespace DebugGlobals
{
	int visited{ 0 };
};

struct Card
{
	Card() : rank{ 0 }, suit{ 0 } {};
	explicit Card(uint8_t card) : rank{ static_cast<uint8_t>(card >> 2) }, suit{ static_cast<uint8_t>(card & 0b11) } {};
	uint8_t rank : 6;
	uint8_t suit : 2;

	operator uint8_t() const {return (rank << 2) | (suit);}
	operator uint8_t() {return (rank << 2) | (suit);}

	operator int() { return (rank << 2) | (suit); }
};


struct Deck
{
	void init()
	{
		for (int i = Constants::nDeckCards - 1; i >= 0; --i)
		{
			deck[i] = Card{ static_cast<uint8_t>(i)};
		}
	}

	void shuffle()
	{
		init();
		for (int i = Constants::nDeckCards - 1; i >= 0; --i)
		{
			int rand = Random::get(0, i);
			std::swap(deck[i], deck[rand]);
		}
	}
	
	std::array<Card, Constants::nFlopCards> getFlopCards() const { return { deck[Constants::communityCardIndex], deck[Constants::communityCardIndex+1], deck[Constants::communityCardIndex+2] }; } 
	std::array<Card, Constants::nTurnCards> getTurnCards() const { return { deck[Constants::communityCardIndex + 3] }; } 
	std::array<Card, Constants::nRiverCards> getRiverCards() const { return { deck[Constants::communityCardIndex + 4] }; } 
	
	std::array<Card, Constants::nPrivateCards> getPreflopHand(int playerIdx) const { return { deck[playerIdx * 2], deck[playerIdx * 2 + 1] }; } 

	std::array<Card, Constants::nPrivateCards + Constants::nFlopCards> getFlopHand(int playerIdx) const
	{
		const auto& preflopCards = getPreflopHand(playerIdx);
		const auto& flopCards = getFlopCards();
		
		std::array<Card, Constants::nPrivateCards + Constants::nFlopCards> flopHand;
		std::copy(preflopCards.begin(), preflopCards.end(), flopHand.begin());
		std::copy(flopCards.begin(), flopCards.end(), flopHand.begin() + Constants::nPrivateCards);

		return flopHand;
	}

	std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards> getTurnHand(int playerIdx) const
	{
		const auto& preflopCards = getPreflopHand(playerIdx);
		const auto& flopCards = getFlopCards();
		const auto& turnCards = getTurnCards();
		
		std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards> turnHand;
		std::copy(preflopCards.begin(), preflopCards.end(), turnHand.begin());
		std::copy(flopCards.begin(), flopCards.end(), turnHand.begin() + Constants::nPrivateCards);
		std::copy(turnCards.begin(), turnCards.end(), turnHand.begin() + Constants::nPrivateCards + Constants::nFlopCards);

		return turnHand;
	}

	using RiverHand = std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards + Constants::nRiverCards>;
	RiverHand getRiverHand(int playerIdx) const
	{
		const auto& preflopCards = getPreflopHand(playerIdx);
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

	std::array<Card, Constants::nDeckCards> deck { };
};

using FoldedPlayers = Flags8Bits<uint8_t>;

inline int getPreviousPlayer(int currentPlayerId)
{
	return currentPlayerId == 0 ? Constants::nPlayers - 1 : currentPlayerId - 1;
}

inline int getNextPlayer(int currentPlayerId)
{
	return (currentPlayerId + 1) % Constants::nPlayers;
}

inline int getSBPlayerIdFromButton(int buttonPlayerId)
{
	return (buttonPlayerId + 1) % Constants::nPlayers;
}

inline int getBBPlayerIdFromButton(int buttonPlayerId)
{
	return (buttonPlayerId + 2) % Constants::nPlayers;
}

struct GameContext
{
	GameContext()
	{
		for (auto& round : history)
		{
			round.fill(Action::Count);
		}
	}

	FoldedPlayers foldedPlayersStatus{ 0 };
	BettingRound currentBettingRound{ BettingRound::Preflop };
	int roundActionIndex {0};
	int nBetInCurrentRound{ 0 };
	std::array<int, Constants::nPlayers> stacks{ 0 };
	std::array<int, Constants::nPlayers> moneyInPot{ 0 };
	Deck deck{ };
	int pot{ 0 };

	int currentButtonPlayerId{ 0 };
	int updatingPlayerId{ 0 };
	int currentPlayerTurn{ 0 };
	int lastBettingPlayer{ 0 };

	int currentBet{ 0 };
	
	// TODO (1)
	// std::string gameHistory{};
	GameHistory history {};

	// NOTE (keb): The BB has checked on the BB Preflop exception.
	bool isCheckAfterBBPreflopException {false};
	// NOTE (keb): When no bets have occured Preflop and it's the BB's turn.
	bool isBBPreflopException() const 
	{ 
		return lastBettingPlayer == getBBPlayerIdFromButton(currentButtonPlayerId) && 
			currentPlayerTurn == getBBPlayerIdFromButton(currentButtonPlayerId) &&
			nBetInCurrentRound == 1 && 
			currentBettingRound == BettingRound::Preflop; 
	}

	bool allinState()
	{
		int nPlayersAllin {0};
		for (int i = 0; i < Constants::nPlayers; ++i)
		{
			if (!foldedPlayersStatus.test(i) && stacks[i] <= 0)
			{
				++nPlayersAllin;
			}
		}
		int nPlayersRemaining = Constants::nPlayers - foldedPlayersStatus.nSet();
		return nPlayersAllin >= nPlayersRemaining-1;
	}

};

class Infoset
{
public:
	struct Key
	{
		// NOTE (keb) 72 bits for at most 6 actions per betting round * 4 betting rounds * 3 bits per action
		__uint128_t history : 72;
		uint16_t clusterIndex : 10;
		uint8_t playerid : 4;			
		uint8_t stacks : 6;
		uint8_t pot : 3;
		
		bool operator==(const Key &other) const
		{ return (history == other.history 
			&& playerid == other.playerid
			&& clusterIndex == other.clusterIndex
			&& stacks == other.stacks
			&& pot == other.pot);
		}
	};

	struct KeyHasher
	{
		std::size_t operator()(const Key& k) const
		{
			using std::size_t;
			using std::hash;
			using std::string;

			std::size_t res = 17;
			res = res * 31 + hash<uint64_t>()( k.history);
			res = res * 31 + hash<uint16_t>()( k.clusterIndex);
			res = res * 31 + hash<uint8_t>()( k.playerid);
			res = res * 31 + hash<uint8_t>()( k.stacks);
			res = res * 31 + hash<uint8_t>()( k.pot);
			return res;
		}
	};

	std::array<int, Action::Count>& getRegrets() { return regrets; }

	Actions getActions() { return allowedActions; }
	void setActions(const GameContext& gameContext) 
	{
		int pot = gameContext.pot;
		int stack = gameContext.stacks[gameContext.currentPlayerTurn];

		if (stack <= 0)
		{
			return;
		}
		if (gameContext.isBBPreflopException())
		{
			allowedActions.set(Check);
		}
		else
		{
			allowedActions.set(Call);
		}
		if (gameContext.nBetInCurrentRound > 0)
		{
			allowedActions.set(Fold);
		}
		if (gameContext.nBetInCurrentRound >= 3)
		{
			return;
		}
		if (stack > 0)
		{
			allowedActions.set(BetQuarter);
		}
		if (stack > 0.25*pot)
		{
			allowedActions.set(BetHalf);
		}
		if (stack > 0.5*pot)
		{
			allowedActions.set(BetPot);
		}
		if (stack > pot)
		{
			allowedActions.set(BetOver);
		}
	}

	const std::array<float, Action::Count> calculateStrategy()
	{
		std::array<float, Action::Count> strategy{ 0.0 };
		int sum = 0;
		for (int a = 0; a < Action::Count; ++a)
		{
			Action action = static_cast<Action>(a);
			if (allowedActions.test(action))
			{
				sum += regrets[a] <= 0 ? 0 : regrets[a];
			}
		}

		int nAllowedActions = __builtin_popcount(allowedActions.flags);
		for (int a = 0; a < Action::Count; ++a)
		{
			Action action = static_cast<Action>(a);
			if (allowedActions.test(action))
			{
				strategy[a] = sum > 0.0 ? ((regrets[a] <= 0 ? 0 : regrets[a]) / static_cast<float>(sum)) : (1.0 / nAllowedActions);
			}
		}

		return strategy;
	}

	void discountRegrets(double discount)
	{
		for (int a = 0; a < Action::Count; ++a)
		{
			Action action = static_cast<Action>(a);
			if (allowedActions.test(action))
			{
				regrets[a] = static_cast<int>(regrets[a] * discount);
			}
		}
	}

	// const std::array<float, Action::Count>& getStrategy() { return strategy; }

private:
	// std::array<float, Action::Count> strategy{ 0.0 };
	std::array<int, Action::Count> regrets{ 0 };
	Actions allowedActions{0};
};

inline std::unordered_map<Infoset::Key, Infoset, Infoset::KeyHasher> infosets;

enum AbstractedSizeRatio : uint8_t
{
	Zero,
	Quarter,
	Half,
	Same,
	Double,
};

std::string toString(AbstractedSizeRatio ratio)
{
	if (ratio == Zero)
		return "Z";
	if (ratio == Quarter)
		return "Q";
	if (ratio == Half)
		return "H";
	if (ratio == Same)
		return "S";
	if (ratio == Double)
		return "D";
	return "?";
}

double toDouble(Action actionSize)
{
	switch (actionSize)
	{
	case BetQuarter:
		return 0.25;
	case BetHalf:
		return 0.5;
	case BetPot:
		return 1.0;
	case BetOver:
		return 3.0;
	default:
		return 0.0;
	}
}


// C.f. [PAPER] doi/abs/10.5555/2540128.2540148
double pseudoHarmonicMapping(double min, double max, double v)
{
	return ((max - v) * (1.0 + min)) / ((max - min) * (1.0 + v));
}

std::array<AbstractedSizeRatio, Constants::nPlayers> abstractStacks(int currentPlayerId, const std::array<int, Constants::nPlayers>& stacks)
{
	// A player's stack is abstracted to be a multiple of the current player's stack.
	std::array<AbstractedSizeRatio, Constants::nPlayers> abstraction;

	const int currentPlayerStack = stacks[currentPlayerId];
	for (int i = 0; i < Constants::nPlayers; ++i)
	{
		double ratio = static_cast<double>(currentPlayerStack) / stacks[i];
		if (ratio >= 2.0)
		{
			abstraction[i] = Double;
		}
		else if (ratio <= 0.5)
		{
			abstraction[i] = Half;
		}
		else
		{
			double probability = 1.0f;
			if (ratio > 0.5 && ratio <= 1.0)
			{
				probability = pseudoHarmonicMapping(0.5, 1.0, ratio);
				abstraction[i] = (Random::rand() <= probability) ? Half : Same;
			}
			else
			{
				probability = pseudoHarmonicMapping(1.0, 2.0, ratio);
				abstraction[i] = (Random::rand() <= probability) ? Same : Double;
			}
		}
	}
	return abstraction;
}

AbstractedSizeRatio abstractPot(int pot, int stack)
{
	// Half, Same, Double, X5
	double ratio = static_cast<double>(pot) / stack;
	if (ratio <= 0.5)
	{
		return Half;
	}
	if (ratio >= 2.0)
	{
		return Double;
	}
	
	double probability = 1.0f;
	if (ratio > 0.5 && ratio <= 1.0)
	{
		probability = pseudoHarmonicMapping(0.5, 1.0, ratio);
		return (Random::rand() <= probability) ? Half : Same;
	}
	else
	{
		probability = pseudoHarmonicMapping(1.0, 2.0, ratio);
		return (Random::rand() <= probability) ? Same : Double;
	}
}

// TODO: horrible code, templatize?
std::string getHandStr(int playerId, const Deck& deck, BettingRound round)
{
	std::string result;
	switch (round)
	{
	case BettingRound::Preflop:
	{
		const auto& hand = deck.getPreflopHand(playerId);
		for (auto card : hand)
		{
			result += Constants::RANK_TO_CHAR[card.rank];
			result += Constants::SUIT_TO_CHAR[card.suit];
		}
		result += '|';
		return result;
	}
	case BettingRound::Flop:
	{
		const auto& hand = deck.getFlopHand(playerId);
		for (auto card : hand)
		{
			result += Constants::RANK_TO_CHAR[card.rank];
			result += Constants::SUIT_TO_CHAR[card.suit];
		}
		result += '|';
		return result;
	}
	case BettingRound::Turn:
	{
		const auto& hand = deck.getTurnHand(playerId);
		for (auto card : hand)
		{
			result += Constants::RANK_TO_CHAR[card.rank];
			result += Constants::SUIT_TO_CHAR[card.suit];
		}
		result += '|';
		return result;
	}
	case BettingRound::River:
	{
		const auto& hand = deck.getRiverHand(playerId);
		for (auto card : hand)
		{
			result += Constants::RANK_TO_CHAR[card.rank];
			result += Constants::SUIT_TO_CHAR[card.suit];
		}
		result += '|';
		return result;
	}
	default:
	{
		return "?";
	}
	}
	return "?";
}

// TODO: horrible code, templatize?
uint16_t getAbstractionIndex(int playerId, const Deck& deck, BettingRound round)
{
	uint64_t result;
	switch (round)
	{
	case BettingRound::Preflop:
	{
		const auto& hand = deck.getPreflopHand(playerId);
		uint8_t cards[Constants::nCardsRoundAccumulator[(uint8_t)BettingRound::Preflop]];
		std::copy(hand.begin(), hand.end(), cards);
		result = hand_index_last(&indexers[0], cards);
		return result;
	}
	case BettingRound::Flop:
	{
		using FlopHistogram = Histogram<AbstractionsContext::nEHSHistogramsBins>;
		const auto& hand = deck.getFlopHand(playerId);
		uint8_t cards[Constants::nCardsRoundAccumulator[(uint8_t)BettingRound::Flop]];
		std::copy(hand.begin(), hand.end(), cards);
		uint64_t handIndex = hand_index_last(&indexers[(uint8_t)round], cards);
		const auto& histogram = flopHistograms[handIndex];
		result = KMeans::findNearestCentroid<FlopHistogram, KMeans::L2Distance<FlopHistogram>>(flopCentroids, histogram).first;
		return result;
	}
	case BettingRound::Turn:
	{
		using TurnHistogram = Histogram<AbstractionsContext::nEHSHistogramsBins>;
		const auto& hand = deck.getTurnHand(playerId);
		uint8_t cards[Constants::nCardsRoundAccumulator[(uint8_t)BettingRound::Turn]];
		std::copy(hand.begin(), hand.end(), cards);
		uint64_t handIndex = hand_index_last(&indexers[(uint8_t)round], cards);
		const auto& histogram = turnHistograms[handIndex];
		result = KMeans::findNearestCentroid<TurnHistogram, KMeans::L2Distance<TurnHistogram>>(turnCentroids, histogram).first;
		return result;
	}
	case BettingRound::River:
	{
		using RiverHistogram = Histogram<AbstractionsContext::nOCHSHistogramsBins>;
		const auto& hand = deck.getRiverHand(playerId);
		uint8_t cards[Constants::nCardsRoundAccumulator[(uint8_t)BettingRound::River]];
		std::copy(hand.begin(), hand.end(), cards);
		uint64_t handIndex = hand_index_last(&indexers[(uint8_t)round], cards);
		const auto& histogram = riverHistograms[handIndex];
		result = KMeans::findNearestCentroid<RiverHistogram, KMeans::L2Distance<RiverHistogram>>(riverCentroids, histogram).first;
		return result;
	}
	default:
	{
		return std::numeric_limits<uint16_t>::max();
	}
	}
	return std::numeric_limits<uint16_t>::max();
}

uint64_t serializeHistory(const GameHistory& history)
{
	uint64_t result {0};
	for (int i = 0; i < (uint8_t)BettingRound::RoundCount; ++i)
	{
		for (int j = 0; j < 6; ++j)
		{
			Action a = history[i][j];
			result |= (uint64_t)a << ((i*6*3)+(j*3));
		}
	}
	return result;
}
	
// NOTE: a game history is encoded like so:
//       actions/actions/actions/actions, where the '/' seperates the betting rounds.
//       An information set key is an encoding of the playerId, the information abstraction index,
//       the game history, the players' stacks relative to the current state player, 
//       the pot size relative to the current state player. 
//		 All are divided by '|', like so: 
//       playerId|clusterIndex|actions/actions/actions|ActivePlayer1Stack/ActivePlayer2Stack|RelativePot
Infoset::Key makeInfosetKey(const GameHistory& history,
	int playerId,
	const Deck& deck,
	BettingRound bettingRound,
	std::array<AbstractedSizeRatio, Constants::nPlayers> stacks, 
	AbstractedSizeRatio pot,
	FoldedPlayers foldedPlayers)
{
	Infoset::Key key;
	uint16_t clusterIndex = getAbstractionIndex(playerId, deck, bettingRound);
	key.clusterIndex = clusterIndex;
	key.playerid = playerId;
	key.history = serializeHistory(history);

	key.stacks = 0;
	for (int i = 0; i < Constants::nPlayers; ++i)
	{
		bool isFolded = foldedPlayers.test(i);
		// NOTE (keb): shift bits by 3 because we reserve 3 bits to represent a stack.
		key.stacks |= (isFolded ? ((uint8_t)Zero << (i*3)) : (uint8_t(stacks[i]) << (i*3)));
	}
	key.pot = (uint8_t)pot;
	
	return key;
}

Action sampleAction(double rand, const std::array<float, Action::Count>& strategy, Actions allowedActions)
{
	float randAcc = 0.0;
	for (int i = 0; i < strategy.size(); ++i)
	{
		if (!allowedActions.test(static_cast<Action>(i)))
		{
			continue;
		}
		randAcc += strategy[i];
		if (rand <= randAcc)
		{
			return static_cast<Action>(i);
		}
	}
	return Action::Count;
}

void call(GameContext& gameContext)
{
	auto& stack = gameContext.stacks[gameContext.currentPlayerTurn];
	int amount = gameContext.currentBet > stack ? stack : gameContext.currentBet;
	stack -= amount;
	gameContext.pot += amount;
	gameContext.moneyInPot[gameContext.currentPlayerTurn] += amount;
}

void undoCall(GameContext& gameContext, int previousStack, int previousBet, int previousPot)
{
	auto& stack = gameContext.stacks[gameContext.currentPlayerTurn];
	int amount = previousBet > previousStack ? previousStack : previousBet;
	stack += amount;
	gameContext.pot -= amount;
	gameContext.moneyInPot[gameContext.currentPlayerTurn] -= amount;
}

void check(GameContext& gameContext)
{
	gameContext.isCheckAfterBBPreflopException = true;
}
void undoCheck(GameContext& gameContext)
{
	gameContext.isCheckAfterBBPreflopException = false;
}

void bet(int playerId, int amount, GameContext& gameContext)
{
	gameContext.moneyInPot[playerId] += amount;
	gameContext.stacks[playerId] -= amount;
	gameContext.pot += amount;
	gameContext.currentBet = amount;
	++gameContext.nBetInCurrentRound;
	gameContext.lastBettingPlayer = playerId;
}

void undoBet(int playerId, int amount, GameContext& gameContext, int previousBet, int previousPot, int previousLastBettingPlayer)
{
	gameContext.moneyInPot[playerId] -= amount;
	gameContext.stacks[playerId] += amount;
	gameContext.pot -= amount;
	gameContext.currentBet = previousBet;
	--gameContext.nBetInCurrentRound;
	gameContext.lastBettingPlayer = previousLastBettingPlayer;
}

void betRelativePot(GameContext& gameContext, double ratio)
{
	int pot = gameContext.pot;
	int stack = gameContext.stacks[gameContext.currentPlayerTurn];
	int amount = stack >= ratio * pot ? static_cast<int>(ratio * pot) : stack;
	bet(gameContext.currentPlayerTurn, amount, gameContext);
}

void undoBetRelativePot(GameContext& gameContext, double ratio, int previousStack, int previousBet, int previousPot, int previousLastBettingPlayer)
{
	int amount = previousStack >= ratio * previousPot ? static_cast<int>(ratio * previousPot) : previousStack;
	undoBet(gameContext.currentPlayerTurn, amount, gameContext, previousBet, previousPot, previousLastBettingPlayer);
}

void applyAction(GameContext& gameContext, Action a)
{
	switch (a)
	{
	case Fold:
	{
		gameContext.foldedPlayersStatus.set(gameContext.currentPlayerTurn);
		break;
	}
	case Call:
	{
		call(gameContext);
		break;
	}
	case Check:
	{
		check(gameContext);
		break;
	}
	case BetQuarter:
	case BetHalf:
	case BetPot:
	case BetOver:
	{
		betRelativePot(gameContext, toDouble(a));
		break;
	}
	default:
	{
		std::cerr << "Applying Invalid Action" << std::endl;
	}
	}
}

void unApplyAction(GameContext& gameContext, Action a, int previousStack, int previousBet, int previousPot, int previousLastBettingPlayer)
{
	switch (a)
	{
	case Fold:
	{
		gameContext.foldedPlayersStatus.unset(gameContext.currentPlayerTurn);
		break;
	}
	case Call:
	{
		undoCall(gameContext, previousStack, previousBet, previousPot);
		break;
	}
	case Check:
	{
		undoCheck(gameContext);
		break;
	}
	case BetQuarter:
	case BetHalf:
	case BetPot:
	case BetOver:
	{
		undoBetRelativePot(gameContext, toDouble(a), previousStack, previousBet, previousPot, previousLastBettingPlayer);
		break;
	}
	default:
	{
		std::cerr << "Unapplying Invalid Action" << std::endl;
	}
	}
}

struct PreviousBettingRoundSnapshot
{
	bool hasSnapped {false};
	// TODO (1)
	// std::string history;
	int playerId;
	int lastBettingPlayerId;
	int currentBet;
	int nBet;
	BettingRound currentBettingRound;
	int roundActionIndex;
};

void snapBettingRound(const GameContext& gameContext, PreviousBettingRoundSnapshot& snapshot)
{
	snapshot.hasSnapped = true;
	// TODO (1)
	// snapshot.history = gameContext.gameHistory;
	snapshot.playerId = gameContext.currentPlayerTurn;
	snapshot.lastBettingPlayerId = gameContext.lastBettingPlayer;
	snapshot.currentBet = gameContext.currentBet;
	snapshot.nBet = gameContext.nBetInCurrentRound;
	snapshot.currentBettingRound = gameContext.currentBettingRound;
	snapshot.roundActionIndex = gameContext.roundActionIndex;
}

void undoBettingRound(const PreviousBettingRoundSnapshot& snapshot, GameContext& gameContext)
{
	// TODO (1)
	// if (snapshot.history.empty())
	if (!snapshot.hasSnapped)
	{
		return;
	}
	// TODO (1)
	// gameContext.gameHistory = snapshot.history;
	gameContext.currentPlayerTurn = snapshot.playerId;
	gameContext.lastBettingPlayer = snapshot.lastBettingPlayerId;
	gameContext.currentBet = snapshot.currentBet;
	gameContext.nBetInCurrentRound = snapshot.nBet;
	gameContext.currentBettingRound = snapshot.currentBettingRound;
	gameContext.roundActionIndex = snapshot.roundActionIndex;
}

float traverseMccfr(GameContext& gameContext)
{
	PreviousBettingRoundSnapshot previousRoundSnapshot;
	// STEP 1: check terminal state
	// Terminal State, Updating Player is not playing
	if (gameContext.foldedPlayersStatus.test(gameContext.updatingPlayerId))
	{
		return -gameContext.moneyInPot[gameContext.updatingPlayerId];
	}
	// Terminal Last One Standing 
	if (gameContext.foldedPlayersStatus.one())
	{
		if (!gameContext.foldedPlayersStatus.test(gameContext.updatingPlayerId))
		{
			return gameContext.pot;
		}
		// TODO: this branch might be useless, as when the updating player folds, the search ends.
		else
		{
			return -gameContext.moneyInPot[gameContext.updatingPlayerId];
		}
	}

	// STEP 2: check if the current betting round is over.
	if ((gameContext.currentPlayerTurn == gameContext.lastBettingPlayer && !(gameContext.isBBPreflopException())) ||
		gameContext.isCheckAfterBBPreflopException ||
		gameContext.allinState())
	{
		gameContext.isCheckAfterBBPreflopException = false;
		if (gameContext.currentBettingRound == BettingRound::River)
		{
			// Showdown
			int bestHandSeen = INT_MAX;
			std::unordered_set<int> tyingPlayerIds{};
			for (int i = 0; i < Constants::nPlayers; ++i)
			{
				if (gameContext.foldedPlayersStatus.test(i))
				{
					continue;
				}
				auto hand = gameContext.deck.getRiverHand(i);
				if (int handStrength = std::apply(evaluate_7cards, hand); handStrength <= bestHandSeen) // smaller index is stronger
				{
					if (handStrength == bestHandSeen)
					{
						tyingPlayerIds.insert(i);
						continue;
					}
					tyingPlayerIds.clear();
					tyingPlayerIds.insert(i);
					bestHandSeen = handStrength;
				}
			}
			if (tyingPlayerIds.find(gameContext.updatingPlayerId) != tyingPlayerIds.end())
			{
				return static_cast<int>(gameContext.pot / tyingPlayerIds.size()); // Integer Division, inaccuracy negligible 
			}
			else
			{
				return -gameContext.moneyInPot[gameContext.updatingPlayerId];
			}
		}
		snapBettingRound(gameContext, previousRoundSnapshot);
		gameContext.currentBettingRound = (BettingRound) ((uint8_t)gameContext.currentBettingRound + 1);
		// TODO (1)
		// gameContext.gameHistory += '/';
		gameContext.currentPlayerTurn = getSBPlayerIdFromButton(gameContext.currentButtonPlayerId); // The SB starts the new round.
		gameContext.lastBettingPlayer = gameContext.currentPlayerTurn;
		gameContext.currentBet = 0;
		gameContext.nBetInCurrentRound = 0;
		gameContext.roundActionIndex = 0;
	}

	float value = 0.0;
	if (gameContext.allinState())
	{
		value = traverseMccfr(gameContext);
	}
	// STEP 3: check if the current player is still in play.
	else if (gameContext.foldedPlayersStatus.test(gameContext.currentPlayerTurn))
	{
		gameContext.currentPlayerTurn = getNextPlayer(gameContext.currentPlayerTurn);
		value = traverseMccfr(gameContext);
		gameContext.currentPlayerTurn = getPreviousPlayer(gameContext.currentPlayerTurn);
	}
	// STEP 4: apply current player's actions.
	else
	{
		auto abstractedStacks = abstractStacks(gameContext.currentPlayerTurn, gameContext.stacks);
		auto abstractedPot = abstractPot(gameContext.pot, gameContext.stacks[gameContext.currentPlayerTurn]);
		Infoset::Key infosetKey = makeInfosetKey(gameContext.history, gameContext.currentPlayerTurn, gameContext.deck, gameContext.currentBettingRound, abstractedStacks, abstractedPot, gameContext.foldedPlayersStatus);
		auto [infosetItr, isNewState] = infosets.try_emplace(infosetKey, Infoset());
		if (isNewState)
		{
			infosetItr->second.setActions(gameContext);
			infosetItr->second.calculateStrategy();
		}


		Infoset& infoset = infosetItr->second;
		if (gameContext.currentPlayerTurn == gameContext.updatingPlayerId)
		{
			++DebugGlobals::visited;
			const auto strategy = infoset.calculateStrategy();
			Actions allowedActions = infoset.getActions();
			std::array<float, Action::Count> actionValues{ 0 };
			for (int i = 0; i < Action::Count; ++i)
			{
				Action a = static_cast<Action>(i);
				if (!allowedActions.test(a))
				{
					continue;
				}
				gameContext.history[(uint8_t)gameContext.currentBettingRound][gameContext.roundActionIndex] = a;
				++gameContext.roundActionIndex;
				int previousStack = gameContext.stacks[gameContext.currentPlayerTurn];
				int previousBet = gameContext.currentBet;
				int previousPot = gameContext.pot;
				int previousLastBettingPlayer = gameContext.lastBettingPlayer;
				applyAction(gameContext, a);
				gameContext.currentPlayerTurn = getNextPlayer(gameContext.currentPlayerTurn);
				actionValues[a] = traverseMccfr(gameContext);
				gameContext.currentPlayerTurn = getPreviousPlayer(gameContext.currentPlayerTurn);
				unApplyAction(gameContext, a, previousStack, previousBet, previousPot, previousLastBettingPlayer);
				value += strategy[a] * actionValues[a];
				--gameContext.roundActionIndex;
				gameContext.history[(uint8_t)gameContext.currentBettingRound][gameContext.roundActionIndex] = Action::Count;
			}
			
			auto& regrets= infoset.getRegrets();
			for (int i = 0; i < Action::Count; ++i)
			{
				Action a = static_cast<Action>(i);
				if (!allowedActions.test(a))
				{
					continue;
				}
				regrets[a] += static_cast<int>(actionValues[a] - value);
			}
			infoset.calculateStrategy();
		}
		else
		{
			const auto strategy = infoset.calculateStrategy();
			Random::rand();
			Action a = sampleAction(Random::rand(), strategy, infoset.getActions());
			gameContext.history[(uint8_t)gameContext.currentBettingRound][gameContext.roundActionIndex] = a;
			++gameContext.roundActionIndex;
			int previousStack = gameContext.stacks[gameContext.currentPlayerTurn];
			int previousBet = gameContext.currentBet;
			int previousPot = gameContext.pot;
			int previousLastBettingPlayer = gameContext.lastBettingPlayer;
			applyAction(gameContext, a);
			gameContext.currentPlayerTurn = getNextPlayer(gameContext.currentPlayerTurn);
			value = traverseMccfr(gameContext);
			gameContext.currentPlayerTurn = getPreviousPlayer(gameContext.currentPlayerTurn);
			unApplyAction(gameContext, a, previousStack, previousBet, previousPot, previousLastBettingPlayer);
			--gameContext.roundActionIndex;
			gameContext.history[(uint8_t)gameContext.currentBettingRound][gameContext.roundActionIndex] = Action::Count;
		}
	}
	
	undoBettingRound(previousRoundSnapshot, gameContext);
	return value;
}

void initAbstractions()
{
	using namespace std::chrono_literals;
	using namespace PokerTypes::AbstractionsContext;
	
	const uint8_t preflop[] = {2};
	const uint8_t flop[] = {2,3};
	const uint8_t turn[] = {2,3,1};
	const uint8_t river[] = {2,3,1,1};
	assert(hand_indexer_init(1, preflop, &indexers[0]));
	assert(hand_indexer_init(2, flop, &indexers[1]));
	assert(hand_indexer_init(3, turn, &indexers[2]));
	assert(hand_indexer_init(4, river, &indexers[3]));
}

void mccfrP(int nIterations)
{
	using namespace std::chrono_literals;
	using namespace PokerTypes::AbstractionsContext;

	infosets.reserve(1'000'000'000);
	
	initAbstractions();
	
	const auto nFlopSize = hand_indexer_size(&indexers[(uint8_t)BettingRound::River], (uint8_t)BettingRound::Flop);
	const auto nTurnSize = hand_indexer_size(&indexers[(uint8_t)BettingRound::River], (uint8_t)BettingRound::Turn);
	const auto nRiverSize = hand_indexer_size(&indexers[(uint8_t)BettingRound::River], (uint8_t)BettingRound::River);

	std::unique_ptr<Histogram<AbstractionsContext::nEHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nEHSHistogramsBins>>> flopHistogramsMmap = Memory::getMmap<Histogram<nEHSHistogramsBins>>((std::string("../EHS/") + flopHistogramsFilename).c_str(), nFlopSize);
	std::unique_ptr<Histogram<AbstractionsContext::nEHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nEHSHistogramsBins>>> turnHistogramsMmap = Memory::getMmap<Histogram<nEHSHistogramsBins>>((std::string("../EHS/") + turnHistogramsFilename).c_str(), nTurnSize);
	std::unique_ptr<Histogram<AbstractionsContext::nOCHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nOCHSHistogramsBins>>> riverHistogramsMmap = Memory::getMmap<Histogram<nOCHSHistogramsBins>>((std::string("../EHS/") + riverOCHSFilename).c_str(), nRiverSize);

	std::unique_ptr<Histogram<AbstractionsContext::nEHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nEHSHistogramsBins>>> flopCentroidsMmap = Memory::getMmap<Histogram<nEHSHistogramsBins>>((std::string("../EHS/") + flopCentroidsFilename).c_str(), nFlopBuckets);
	std::unique_ptr<Histogram<AbstractionsContext::nEHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nEHSHistogramsBins>>> turnCentroidsMmap = Memory::getMmap<Histogram<nEHSHistogramsBins>>((std::string("../EHS/") + turnCentroidsFilename).c_str(), nTurnBuckets);
	std::unique_ptr<Histogram<AbstractionsContext::nOCHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nOCHSHistogramsBins>>> riverCentroidsMmap = Memory::getMmap<Histogram<nOCHSHistogramsBins>>((std::string("../EHS/") + riverCentroidsFilename).c_str(), nRiverBuckets);
	
	flopHistograms = {flopHistogramsMmap.get(), flopHistogramsMmap.get() + nFlopSize};
	turnHistograms = {turnHistogramsMmap.get(), turnHistogramsMmap.get() + nTurnSize};
	riverHistograms = {riverHistogramsMmap.get(), riverHistogramsMmap.get() + nRiverSize};
	
	flopCentroids = {flopCentroidsMmap.get(), flopCentroidsMmap.get() + nFlopBuckets};
	turnCentroids = {turnCentroidsMmap.get(), turnCentroidsMmap.get() + nTurnBuckets};
	riverCentroids = {riverCentroidsMmap.get(), riverCentroidsMmap.get() + nRiverBuckets};
		
	startTime = std::chrono::steady_clock::now();

	for (int t = 0; t < nIterations; ++t)
	{
		GameContext gameContext{};
		// Setup starting stacks
		for (auto& stack : gameContext.stacks)
		{
			stack = Random::get(Constants::minStartingStackSize, Constants::maxStartingStackSize);
		}

		gameContext.deck.shuffle();

		gameContext.currentButtonPlayerId = 0;

		const auto timeElapsed = Time::getTimeElapsed(startTime, 1min);
		for (int playerId = 0; playerId < Constants::nPlayers; ++playerId)
		{
			// TODO (keb): Take a snapshot of the current strategy to make an average of the snapshots later.
			//if (t % Constants::strategyInterval == 0)
			//{
			//	updateStrategy(_, playerIdx);
			//}

			// if (timeElapsed > Constants::pruningThresholdMinutes)
			// {
			// 	const bool isPruningIteration = Random::rand() < 0.95;
			// 	isPruningIteration ? traverseMccfrPruning(_, _) : traverseMccfr(_, _);
			// }
			// else
			{
				gameContext.currentBettingRound = BettingRound::Preflop;
				gameContext.updatingPlayerId = playerId;
				gameContext.currentPlayerTurn = getNextPlayer(getBBPlayerIdFromButton(gameContext.currentButtonPlayerId)); // The player following the Big Blind.
				bet(getSBPlayerIdFromButton(gameContext.currentButtonPlayerId), Constants::sb, gameContext);
				bet(getBBPlayerIdFromButton(gameContext.currentButtonPlayerId), Constants::bb, gameContext);
				gameContext.nBetInCurrentRound = 1;
				traverseMccfr(gameContext);
				undoBet(getBBPlayerIdFromButton(gameContext.currentButtonPlayerId), Constants::bb, gameContext, 0, 0, 0);
				undoBet(getSBPlayerIdFromButton(gameContext.currentButtonPlayerId), Constants::sb, gameContext, 0, 0, 0);
			}
		}
		
		if (timeElapsed < Constants::LCFRThresholdMinutes && 
			(timeElapsed % Constants::discountIntervalMinutes == 0))
		{
			double discount = (timeElapsed / static_cast<double>(Constants::discountIntervalMinutes)) / ((timeElapsed / static_cast<double>(Constants::discountIntervalMinutes)) + 1);
			for (auto& [key, infoset] : infosets)
			{
				infoset.discountRegrets(discount);
			}
		}
	}
}
};

int main()
{
	MCCFR::mccfrP(100000);
	return 0;
}