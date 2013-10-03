#ifndef CLREGION_H_INCLUDED
#define CLREGION_H_INCLUDED

#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <CL/cl.hpp>

std::string getCLError(cl_int err);

#include "clspatial.h"
#include "cltemporal.h"

struct CLStats
{
	// Spatial pooler
	double averageBoost;

	// Temporal pooler
	int totalSegments;
	int maxSegments;
	int totalSynapses;
	int maxSynapses;
	int averageDutyCycle;
};

class CLRegion
{
private:
	cl::CommandQueue m_commandQueue;

	CLSpatialPooler m_spatialPooler;
	CLTemporalPooler m_temporalPooler;

public:

	CLRegion(cl::Device& device, cl::Context& context, int columns, int inputSize)
		: m_commandQueue(context, device)
		, m_spatialPooler(device, context, m_commandQueue, columns, inputSize)
		, m_temporalPooler(device, context, m_commandQueue, columns)
	{
		std::cerr << "Device memory allocation limit: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
	};

	CLRegion(const CLRegion&) = delete;
	CLRegion(CLRegion&&) = default;

	void write(std::vector<cl_char>& activations, std::vector<cl_char>& results);

	CLStats getStats();
};

#endif
