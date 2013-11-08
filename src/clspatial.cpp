#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include <random>
#include <sstream>

#include "clregion.h"

constexpr static const char* SPATIAL_SRC =
#include "spatial.cl.h"
;

CLSpatialPooler::CLSpatialPooler(CLContext& context, const CLTopology& topo, const CLArgs& args)
	: m_context(context)
	, m_topology(topo)
	, m_args(args)
	, m_columnData(context, m_topology.getColumns())
	, m_synapseData(context, m_topology.getColumns() * args.ColumnProximalSynapseCount)
	, m_inputData(context, m_topology.getInputSize())
	, m_refineCounter(0)
{
	std::cerr << "CLSpatialPooler: Initializing" << std::endl;

	// Install kernel programs
	std::string definitions = args.serialize() + topo.serialize();

	cl::Program::Sources sources;
	sources.push_back({definitions.c_str(), definitions.length()});
	sources.push_back({SPATIAL_SRC, strlen(SPATIAL_SRC)});

	cl::Program program(context.nativeContext(), sources);
	try
	{
		program.build({context.device()});
	}
	catch(const cl::Error& err)
	{
		std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.device()) << std::endl;
		throw;
	}

	m_computeOverlapKernel = cl::KernelFunctor(cl::Kernel(program, "computeOverlap"), context.queue(), cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_inhibitNeighboursKernel = cl::KernelFunctor(cl::Kernel(program, "inhibitNeighbours"), context.queue(), cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_updatePermanencesKernel = cl::KernelFunctor(cl::Kernel(program, "updatePermanences"), context.queue(), cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_refineRegionKernel = cl::KernelFunctor(cl::Kernel(program, "refineRegion"), context.queue(), cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);

	// Initialize region
	cl::KernelFunctor initRegion =
	cl::KernelFunctor(cl::Kernel(program, "initRegion"), context.queue(),
		cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);

	cl_uint2 randomState;
	randomState.s[0] = rand();
	randomState.s[1] = rand();
	initRegion(m_columnData.buffer(), m_synapseData.buffer(), randomState);

	std::cerr << "CLSpatialPooler: Kernels loaded" << std::endl;
}
std::vector<cl_char> CLSpatialPooler::write(const std::vector<cl_char>& bits)
{
	if (bits.size() != std::size_t(m_topology.getInputSize()))
	{
		throw std::runtime_error("Invalid vector length!");
	}

	// Send given input pattern to compute device
	m_inputData.enqueueWrite(false, bits);

	// Phase 1: Overlap
	m_computeOverlapKernel(m_columnData.buffer(), m_synapseData.buffer(), m_inputData.buffer());

	// Phase 2: Inhibit neighbours
	m_inhibitNeighboursKernel(m_columnData.buffer(), m_synapseData.buffer());

	// Phase 3: Update permanences
	m_updatePermanencesKernel(m_columnData.buffer(), m_synapseData.buffer(), m_inputData.buffer());

	// Extra: Refine region (reset bad synapses) every N iterations
	if (++m_refineCounter > 100)
	{
		cl_uint2 randomState;
		randomState.s[0] = rand();
		randomState.s[1] = rand();
		m_refineRegionKernel(m_columnData.buffer(), m_synapseData.buffer(), randomState);
		m_refineCounter = 0;
	}

	// Download list of active columns from the compute device
	m_columnData.enqueueRead(true);

	std::vector<cl_char> ret;
	ret.reserve(m_topology.getColumns());
	for (CLColumn& col: m_columnData)
		ret.push_back(col.active);
	return ret;
}
void CLSpatialPooler::getStats(CLStats& stats)
{
	m_columnData.enqueueRead(true);

	stats.averageBoost = 0;
	stats.averageDutyCycle = 0;

	for (CLColumn& col: m_columnData)
	{
		stats.averageBoost += col.boost;
		stats.averageDutyCycle += col.activeDutyCycle;
	}
	stats.averageBoost /= m_topology.getColumns();
	stats.averageDutyCycle /= m_topology.getColumns();
}
void CLSpatialPooler::backwards(const std::vector< cl_char >& columnActivation, std::vector< double >& result)
{
	// Make sure we're using the latest model by pulling it from the gpu..
	m_synapseData.enqueueRead(true);

	result.assign(m_topology.getInputSize(), 0);

	for (int i = 0 ; i < m_topology.getColumns(); ++i)
	{
		if (columnActivation[i])
		{
			int index = m_args.ColumnProximalSynapseCount * i;
			for (int a = 0; a < m_args.ColumnProximalSynapseCount; ++a)
			{
				CLSynapse& syn = m_synapseData[index+a];
				if (syn.permanence >= m_args.ConnectedPermanence)
				{
					result[syn.target] += 1;
				}
			}
		}
	}
}

