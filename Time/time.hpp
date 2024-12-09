#pragma once

#include <chrono>

namespace Time
{
	using namespace std::chrono_literals;

	template <typename TClock>
	concept Clock = requires ()
	{
		{ TClock::now() } -> std::convertible_to<std::chrono::time_point<TClock>>;
	};

	template<typename TUnits = std::chrono::seconds, typename TClock = std::chrono::steady_clock>
	requires Clock<TClock>
	inline long long getTimeElapsed(std::chrono::time_point<TClock> start, TUnits units = 1s)
	{
		return (TClock::now() - start) / units;
	}

	template<typename TUnits = std::chrono::seconds, typename TClock = std::chrono::steady_clock>
	requires Clock<TClock>
	inline long long timeRemaining(size_t nElements, size_t currentElement, std::chrono::time_point<TClock> start, TUnits units = 1s)
	{
		auto timeElapsed = getTimeElapsed(start, units);
		return currentElement == 0 ? 
			std::numeric_limits<long long>::max() : 
			((timeElapsed * nElements) / (double)currentElement) - timeElapsed;
	}
	
	template<typename TUnits = std::chrono::seconds, typename TClock = std::chrono::steady_clock>
	requires Clock<TClock>
	inline std::chrono::time_point<TClock> now()
	{
		return TClock::now();
	}
};
