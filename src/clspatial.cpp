#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include <random>

#include "clregion.h"

CLSpatialPooler::CLSpatialPooler(cl::Device& device, cl::Context& context, cl::CommandQueue& queue, int columns, int inputSize)
	: m_device(device)
	, m_context(context)
	, m_commandQueue(queue)
	, m_columns(columns)
	, m_inputSize(inputSize)
	, m_columnDataBuffer(context, CL_MEM_READ_WRITE, columns * sizeof(CLColumn))
	, m_inputDataBuffer(context, CL_MEM_READ_WRITE, inputSize * sizeof(cl_char))
{
	std::cerr << "CLSpatialPooler: Initializing" << std::endl;

	// Install kernel programs
	std::ifstream fin("cl/spatial.cl");
	std::string source{std::istreambuf_iterator<char>(fin),std::istreambuf_iterator<char>()};
	cl::Program::Sources sources;
	sources.push_back({source.c_str(), source.length()});

	cl::Program program(context, sources);
	if (program.build({device}) != CL_SUCCESS)
	{
		std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		throw std::runtime_error("Error compiling OpenCL source!");
	}

	m_computeOverlapKernel = cl::KernelFunctor(cl::Kernel(program, "computeOverlap"), m_commandQueue, cl::NullRange, cl::NDRange(columns), cl::NullRange);
	m_inhibitNeighboursKernel = cl::KernelFunctor(cl::Kernel(program, "inhibitNeighbours"), m_commandQueue, cl::NullRange, cl::NDRange(columns), cl::NullRange);
	m_updatePermanencesKernel = cl::KernelFunctor(cl::Kernel(program, "updatePermanences"), m_commandQueue, cl::NullRange, cl::NDRange(columns), cl::NullRange);

	// Initialize all columns
	m_columnData.resize(columns);

	std::random_device dev;
	std::mt19937 gen(dev());

	for (CLColumn& col: m_columnData)
	{
		col.boost = 1.0;
		col.overlap = 0;
		col.active = false;
		col.activeDutyCycle = 0.1;
		col.minDutyCycle = 0.1;
		col.overlapDutyCycle = 0.1;

		for (CLSynapse& syn: col.proximalSynapses)
		{
			syn.permanence = std::normal_distribution<>(0.2, 0.2)(gen);
			syn.target = std::uniform_int_distribution<>(0, inputSize-1)(gen);
		}
	}

	// Upload columns to GPU
	m_commandQueue.enqueueWriteBuffer(m_columnDataBuffer, CL_TRUE, 0, m_columns * sizeof(CLColumn), &m_columnData[0]);

	std::cerr << "CLSpatialPooler: Kernels loaded" << std::endl;
}
std::vector<cl_char> CLSpatialPooler::write(const std::vector<cl_char>& bits)
{
	if (bits.size() != std::size_t(m_inputSize))
	{
		throw std::runtime_error("Invalid vector length!");
	}

	cl_int err;

	err = m_commandQueue.enqueueWriteBuffer(m_inputDataBuffer, CL_FALSE, 0, m_inputSize * sizeof(cl_char), &bits[0]);
	if (err != CL_SUCCESS)
		throw std::runtime_error(getCLError(err));

	// Phase 1: Overlap
	m_computeOverlapKernel(m_columnDataBuffer, m_inputDataBuffer, m_inputSize);
	err = m_computeOverlapKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error(getCLError(err));

	// Phase 2: Inhibit neighbours
	//  float sparsityTarget, int nWidth, int nHeight, int regionWidth, int regionHeight
	m_inhibitNeighboursKernel(m_columnDataBuffer, 0.04f, 16, 1, m_columns, 1);
	err = m_inhibitNeighboursKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error(getCLError(err));

	// Phase 3: Update permanences
	m_updatePermanencesKernel(m_columnDataBuffer);
	err = m_updatePermanencesKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error(getCLError(err));

	std::vector<cl_char> ret;
	ret.reserve(m_columns);
	m_commandQueue.enqueueReadBuffer(m_columnDataBuffer,CL_TRUE,0,sizeof(CLColumn)*m_columns, &m_columnData[0]);

	for (CLColumn& col: m_columnData)
		ret.push_back(col.active);

	return ret;
}
void CLSpatialPooler::getStats(CLStats& stats)
{
	m_commandQueue.enqueueReadBuffer(m_columnDataBuffer,CL_TRUE,0,sizeof(CLColumn)*m_columns, &m_columnData[0]);

	stats.averageBoost = 0;
	stats.averageDutyCycle = 0;

	for (CLColumn& col: m_columnData)
	{
		stats.averageBoost += col.boost;
		stats.averageDutyCycle += col.activeDutyCycle;
	}
	stats.averageBoost /= m_columns;
	stats.averageDutyCycle /= m_columns;
}
