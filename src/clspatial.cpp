#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include <random>
#include <sstream>

#include "clregion.h"

CLSpatialPooler::CLSpatialPooler(cl::Device& device, cl::Context& context, cl::CommandQueue& queue, const CLTopology& topo, const CLArgs& args)
	: m_device(device)
	, m_context(context)
	, m_commandQueue(queue)
	, m_topology(topo)
	, m_args(args)
	, m_columnDataBuffer(context, CL_MEM_READ_WRITE, m_topology.getColumns() * sizeof(CLColumn))
	, m_synapseDataBuffer(context, CL_MEM_READ_WRITE, m_topology.getColumns() * args.ColumnProximalSynapseCount* sizeof(CLSynapse))
	, m_inputDataBuffer(context, CL_MEM_READ_WRITE, m_topology.getInputSize() * sizeof(cl_char))
{
	std::cerr << "CLSpatialPooler: Initializing" << std::endl;

	// Install kernel programs
	std::ifstream fin("cl/spatial.cl");
	std::string definitions = args.serialize();
	std::string source = std::string{std::istreambuf_iterator<char>(fin),std::istreambuf_iterator<char>()};
	cl::Program::Sources sources;
	sources.push_back({definitions.c_str(), definitions.length()});
	sources.push_back({source.c_str(), source.length()});

	cl::Program program(context, sources);
	if (program.build({device}) != CL_SUCCESS)
	{
		std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		throw std::runtime_error("Error compiling OpenCL source!");
	}

	m_computeOverlapKernel = cl::KernelFunctor(cl::Kernel(program, "computeOverlap"), m_commandQueue, cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_inhibitNeighboursKernel = cl::KernelFunctor(cl::Kernel(program, "inhibitNeighbours"), m_commandQueue, cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_updatePermanencesKernel = cl::KernelFunctor(cl::Kernel(program, "updatePermanences"), m_commandQueue, cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);

	// Initialize all columns
	m_columnData.resize(m_topology.getColumns());
	m_synapseData.resize(m_topology.getColumns() * m_args.ColumnProximalSynapseCount);

	std::random_device dev;
	std::mt19937 gen(dev());

	int synapseIdx = 0;

	for (std::size_t a = 0; a < m_columnData.size(); ++a)
	{
		CLColumn& col = m_columnData[a];
		col.boost = 1.0;
		col.overlap = 0;
		col.active = false;
		col.activeDutyCycle = 0.1;
		col.minDutyCycle = 0.1;
		col.overlapDutyCycle = 0.1;

		for (int i = 0; i < m_args.ColumnProximalSynapseCount; ++i)
		{
			CLSynapse& syn = m_synapseData[i + synapseIdx];
			syn.permanence = std::normal_distribution<>(0.2, 0.2)(gen);
			if (syn.permanence < 0) syn.permanence = 0;
			if (syn.permanence > 1) syn.permanence = 1;

			if (topo.receptiveFieldRadius < 0)
				syn.target = std::uniform_int_distribution<>(0, m_topology.getInputSize()-1)(gen);
			else
			{
				// Map column xy to input space xy
				int ix = std::floor(m_topology.inputWidth * (float(a % m_topology.regionWidth)/m_topology.regionWidth));
				int iy = std::floor(m_topology.inputHeight * (float(a / m_topology.regionWidth)/m_topology.regionHeight));

				int minx = std::max(0,               ix - topo.receptiveFieldRadius);
				int maxx = std::min(topo.inputWidth, ix + topo.receptiveFieldRadius);
				int miny = std::max(0,                iy - topo.receptiveFieldRadius);
				int maxy = std::min(topo.inputHeight, iy + topo.receptiveFieldRadius);

				int rx = std::uniform_int_distribution<>(minx, maxx-1)(gen);
				int ry = std::uniform_int_distribution<>(miny, maxy-1)(gen);

				syn.target = rx + ry*topo.inputWidth;
			}
		}
		synapseIdx += m_args.ColumnProximalSynapseCount;
	}

	// Upload columns to GPU
	m_commandQueue.enqueueWriteBuffer(m_columnDataBuffer, CL_TRUE, 0, sizeof(CLColumn) * m_columnData.size(), &m_columnData[0]);
	m_commandQueue.enqueueWriteBuffer(m_synapseDataBuffer, CL_TRUE, 0, sizeof(CLSynapse) * m_synapseData.size(), &m_synapseData[0]);

	std::cerr << "CLSpatialPooler: Kernels loaded" << std::endl;
}
std::vector<cl_char> CLSpatialPooler::write(const std::vector<cl_char>& bits)
{
	if (bits.size() != std::size_t(m_topology.getInputSize()))
	{
		throw std::runtime_error("Invalid vector length!");
	}

	cl_int err;

	err = m_commandQueue.enqueueWriteBuffer(m_inputDataBuffer, CL_FALSE, 0, m_topology.getInputSize() * sizeof(cl_char), &bits[0]);
	if (err != CL_SUCCESS)
		throw std::runtime_error(getCLError(err));

	// Phase 1: Overlap
	m_computeOverlapKernel(m_columnDataBuffer, m_synapseDataBuffer, m_inputDataBuffer, m_topology.getInputSize());
	err = m_computeOverlapKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error(getCLError(err));

	// Phase 2: Inhibit neighbours
	//  float sparsityTarget, int nWidth, int nHeight, int regionWidth, int regionHeight
	m_inhibitNeighboursKernel(m_columnDataBuffer, m_synapseDataBuffer, 0.04f, m_topology.inhibitionRadius, m_topology.inhibitionRadius, m_topology.regionWidth, m_topology.regionHeight);
	err = m_inhibitNeighboursKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error(getCLError(err));

	// Phase 3: Update permanences
	m_updatePermanencesKernel(m_columnDataBuffer, m_synapseDataBuffer, m_inputDataBuffer);
	err = m_updatePermanencesKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error(getCLError(err));

	std::vector<cl_char> ret;
	ret.reserve(m_topology.getColumns());
	m_commandQueue.enqueueReadBuffer(m_columnDataBuffer, CL_TRUE, 0, sizeof(CLColumn) * m_columnData.size(), &m_columnData[0]);

	for (CLColumn& col: m_columnData)
		ret.push_back(col.active);

	return ret;
}
void CLSpatialPooler::getStats(CLStats& stats)
{
	m_commandQueue.enqueueReadBuffer(m_columnDataBuffer, CL_TRUE, 0, sizeof(CLColumn) * m_columnData.size(), &m_columnData[0]);

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
	m_commandQueue.enqueueReadBuffer(m_synapseDataBuffer, CL_TRUE, 0, sizeof(CLSynapse) * m_synapseData.size() , &m_synapseData[0]);

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

