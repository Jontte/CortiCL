#ifndef CLSPATIAL_H_INCLUDED
#define CLSPATIAL_H_INCLUDED

#define __CL_ENABLE_EXCEPTIONS
#include <string>
#include <CL/cl.hpp>
#include "cltopology.h"
#include "clargs.h"

std::string getCLError(cl_int err);

struct CLStats;
class CLSpatialPooler
{
private:

	struct CLSynapse
	{
		cl_float permanence;
		cl_int target;
	};
	struct CLColumn
	{
		cl_float boost;
		cl_float overlap;
		cl_bool active;
		cl_float activeDutyCycle;
		cl_float minDutyCycle;
		cl_float overlapDutyCycle;
	};

	cl::Device& m_device;
	cl::Context& m_context;
	cl::CommandQueue& m_commandQueue;

	const CLTopology m_topology;
	const CLArgs m_args;

	cl::KernelFunctor m_computeOverlapKernel;
	cl::KernelFunctor m_inhibitNeighboursKernel;
	cl::KernelFunctor m_updatePermanencesKernel;
	cl::KernelFunctor m_refineRegionKernel;

	std::vector<CLColumn> m_columnData;
	std::vector<CLSynapse> m_synapseData;
	cl::Buffer m_columnDataBuffer;
	cl::Buffer m_synapseDataBuffer;
	cl::Buffer m_inputDataBuffer;
	
	int m_refineCounter;

public:

	CLSpatialPooler(cl::Device& device, cl::Context& context, cl::CommandQueue& queue, const CLTopology& topo, const CLArgs& args);
	std::vector<cl_char> write(const std::vector< cl_char >& bits);
	void backwards(const std::vector<cl_char>& columnActivation, std::vector<double>& result);
	void getStats(CLStats& stats);
};


#endif
