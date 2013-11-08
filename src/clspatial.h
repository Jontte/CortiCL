#ifndef CLSPATIAL_H_INCLUDED
#define CLSPATIAL_H_INCLUDED

#include <string>
#include "clcontext.h"
#include "clbuffer.h"
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

	CLContext& m_context;

	const CLTopology m_topology;
	const CLArgs m_args;

	cl::KernelFunctor m_computeOverlapKernel;
	cl::KernelFunctor m_inhibitNeighboursKernel;
	cl::KernelFunctor m_updatePermanencesKernel;
	cl::KernelFunctor m_refineRegionKernel;

	CLBuffer<CLColumn> m_columnData;
	CLBuffer<CLSynapse> m_synapseData;
	CLBuffer<cl_char> m_inputData;

	int m_refineCounter;

public:

	CLSpatialPooler(CLContext& context, const CLTopology& topo, const CLArgs& args);
	std::vector<cl_char> write(const std::vector< cl_char >& bits);
	void backwards(const std::vector<cl_char>& columnActivation, std::vector<double>& result);
	void getStats(CLStats& stats);
};


#endif
