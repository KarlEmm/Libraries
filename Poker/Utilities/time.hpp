#pragma once

#include <chrono>

namespace Time
{
	template <typename TClock>
	concept Clock = requires ()
	{
		{ TClock::now() } -> std::convertible_to<std::chrono::time_point<TClock>>;
	};

	template<typename TUnits, typename TClock = std::chrono::steady_clock>
	requires Clock<TClock>
	inline long long getTimeElapsed(std::chrono::time_point<TClock> start, TUnits units)
	{
		return (TClock::now() - start) / units;
	}
};
