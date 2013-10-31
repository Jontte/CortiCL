#include "sensor.h"

#include <iostream>
#include <algorithm>
#include <cassert>

void Sensor::compute_histogram()
{
	// create histogram

	m_histogram.clear();
	std::sort(m_samples.begin(), m_samples.end());

	std::size_t effectiveSize = m_totalSize - m_windowSize + 1;
	if (effectiveSize <= 0)
		throw std::runtime_error("Sensor window size is too large!");

	size_t stepSize = m_samples.size() / effectiveSize;

	for (size_t i = 0; i < m_samples.size(); i += stepSize)
		m_histogram.push_back(m_samples[i]);

	while (m_histogram.size() > effectiveSize)
		m_histogram.pop_back();
	while (m_histogram.size() < effectiveSize)
		m_histogram.push_back(m_samples.back());

	assert(m_histogram.size() == effectiveSize);
}

std::vector<signed char> Sensor::encode(double value)
{
	std::vector<signed char> ret(m_totalSize, 0);

	if (m_histogram.empty() )
	{
		if (m_samples.size() < SAMPLE_WINDOW_SIZE)
		{
			m_samples.push_back(value);
		}
		else
		{
			compute_histogram();
		}
	}
	else
	{
		size_t counter = 0;
		for (size_t i = 0 ; i < m_histogram.size(); ++i)
		{
			ret[counter++] = value <= m_histogram[i];
			if (ret[counter-1])
			{
				for(int a = 0; a < m_windowSize-1; ++a)
					ret[counter++] = true;
				return ret;
			}
		}
		for(int a = m_totalSize - m_windowSize; a < m_totalSize; ++a)
			ret[a] = true;
	}
	return ret;
}
double Sensor::decode(const std::vector<double>& sdr)
{
	if (m_histogram.empty()) return 0;
	double highestOverlap = -1;
	int highestOverlapIndex = -1;

	for (int i = 0; i+m_windowSize < int(sdr.size()); ++i)
	{
		double overlap = 0;
		for (int a = 0 ; a < m_windowSize; ++a)
		{
			overlap += sdr[i+a];
		}
		if (overlap > highestOverlap)
		{
			highestOverlap = overlap;
			highestOverlapIndex = i;
		}
	}
	return m_histogram[highestOverlapIndex];
}
