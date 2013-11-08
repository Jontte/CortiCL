#include "clcontext.h"
#include <stdexcept>

CLContext::CLContext()
{
	std::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);
	if (platformList.empty())
		throw std::runtime_error("No OpenCL platforms available");

	auto& defaultPlatform = platformList.front();
	std::vector< cl::Device > deviceList;
	defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &deviceList);
	if (deviceList.empty())
		throw std::runtime_error("OpenCL platform contains no devices");

	m_device = deviceList.front();
	m_context = cl::Context({m_device});
	m_queue = cl::CommandQueue(m_context, m_device);
}
