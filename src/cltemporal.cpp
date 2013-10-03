#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include <random>
#include <algorithm>

#include "clregion.h"

CLTemporalPooler::CLTemporalPooler(cl::Device& device, cl::Context& context, cl::CommandQueue& queue, int columns)
	: m_device(device)
	, m_context(context)
	, m_commandQueue(queue)
	, m_columns(columns)
	, m_columnDataBuffer(context, CL_MEM_READ_WRITE, columns * sizeof(CLColumn))
	, m_inputDataBuffer(context, CL_MEM_READ_WRITE, columns * sizeof(cl_char))
{
	std::cerr << "CLTemporalPooler: Initializing" << std::endl;
	// Install kernel programs
	std::ifstream fin("cl/temporal.cl");
	std::string source{std::istreambuf_iterator<char>(fin),std::istreambuf_iterator<char>()};
	cl::Program::Sources sources;
	sources.push_back({source.c_str(), source.length()});

	cl::Program program(context, sources);
	if (program.build({device}) != CL_SUCCESS)
	{
		std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		throw std::runtime_error("Error compiling OpenCL source!");
	}

	m_timeStepKernel = cl::KernelFunctor(cl::Kernel(program, "timeStep"), m_commandQueue, cl::NullRange, cl::NDRange(columns), cl::NullRange);
	m_computeActiveStateKernel = cl::KernelFunctor(cl::Kernel(program, "computeActiveState"), m_commandQueue, cl::NullRange, cl::NDRange(columns), cl::NullRange);
	m_computePredictiveState = cl::KernelFunctor(cl::Kernel(program, "computePredictiveState"), m_commandQueue, cl::NullRange, cl::NDRange(columns), cl::NullRange);
	m_updateSynapsesKernel = cl::KernelFunctor(cl::Kernel(program, "updateSynapses"), m_commandQueue, cl::NullRange, cl::NDRange(columns), cl::NullRange);

	// Initialize all columns
	m_columnData.resize(columns);

	std::random_device dev;
	std::mt19937 gen(dev());

	for (CLColumn& col: m_columnData)
	{
		for (CLCell& cell: col.cells)
		{
			for (CLSegment& seg: cell.segments)
			{
				for (CLSynapse& syn: seg.synapses)
				{
					syn.permanence = 0;
					syn.targetCell = 0;
					syn.targetColumn = 0;
				}
				seg.sequenceSegment = false;
				seg.hasQueuedChanges = false;
			}
			cell.segmentCount = 0;
			cell.state = 0;
		}
	}

	// Upload columns to GPU
	m_commandQueue.enqueueWriteBuffer(m_columnDataBuffer, CL_TRUE, 0, m_columns * sizeof(CLColumn), &m_columnData[0]);
	std::cerr << "CLTemporalPooler: Kernels loaded" << std::endl;
}
void CLTemporalPooler::write(const std::vector< cl_char >& activations_in, std::vector< cl_char >& results_out)
{
	if (activations_in.size() != std::size_t(m_columns))
	{
		throw std::runtime_error("Invalid vector length!");
	}

	cl_int err;

	err = m_commandQueue.enqueueWriteBuffer(m_inputDataBuffer, CL_FALSE, 0, m_columns * sizeof(cl_char), &activations_in[0]);
	if (err != CL_SUCCESS)
		throw std::runtime_error(getCLError(err));

	cl_uint2 randomSeed; // provide gpu some poor man's randomness
	randomSeed.s[0] = rand();
	randomSeed.s[1] = rand();

	// Phase 0: Step forwards in time
	m_timeStepKernel(m_columnDataBuffer);
	err = m_timeStepKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error("timeStepKernel: " + getCLError(err));

	// Phase 1: Compute active state for each cell
	m_computeActiveStateKernel(m_columnDataBuffer, m_inputDataBuffer, randomSeed);
	err = m_computeActiveStateKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error("computeActiveStateKernel: " + getCLError(err));

	// Phase 2: Compute predictive state for each cell
	m_computePredictiveState(m_columnDataBuffer, m_inputDataBuffer, randomSeed);
	err = m_computePredictiveState.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error("computePredictiveStateKernel: " + getCLError(err));

	// Phase 3: Update permanences
	m_updateSynapsesKernel(m_columnDataBuffer, m_inputDataBuffer);
	err = m_updateSynapsesKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error("updateSynapsesKernel: " + getCLError(err));

	results_out.resize(m_columns);
	m_commandQueue.enqueueReadBuffer(m_inputDataBuffer, CL_TRUE, 0, sizeof(cl_char) * m_columns, &results_out[0]);
}

void CLTemporalPooler::getStats(CLStats& stats)
{
	m_commandQueue.enqueueReadBuffer(m_columnDataBuffer, CL_TRUE, 0, sizeof(CLColumn)*m_columns, &m_columnData[0]);

	stats.maxSegments = m_columns * MaxSegments;
	stats.maxSynapses = m_columns * MaxSegments * MaxSynapses;
	stats.totalSegments = 0;
	stats.totalSynapses = 0;
	for (CLColumn& col: m_columnData)
	{
		for (CLCell& cell: col.cells)
		{
			stats.totalSegments += cell.segmentCount;

			for (int i = 0; i < cell.segmentCount; ++i)
			{
				CLSegment& seg = cell.segments[i];
				stats.totalSynapses += seg.synapseCount;
			}
		}
	}
}
