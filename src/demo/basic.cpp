#include <stdexcept>
#include "../clregion.h"

int main()
{
	// Create OpenCL context
	std::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);
	if (platformList.empty())
		throw std::runtime_error("No OpenCL platforms available");

	auto& defaultPlatform = platformList.front();
	std::vector< cl::Device > deviceList;
	defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &deviceList);
	if (deviceList.empty())
		throw std::runtime_error("OpenCL platform contains no devices");

	auto& defaultDevice = deviceList.front();
	cl::Context context({defaultDevice});
	
	// Create region
	
	int inputWidth = 80;
	int inputHeight = 1;
	
	int regionWidth = 80;
	int regionHeight = 1;
	
	int inhibitionRadius = 20;
	int receptiveFieldRadius = 20;
	
	CLRegion region(defaultDevice, context,
		CLTopology::localInhibition2D(inputWidth, inputHeight, regionWidth, regionHeight, inhibitionRadius, receptiveFieldRadius),
		CLArgs()
	);
	
	while (true)
	{
		// Create random input
		std::vector<cl_char> input(inputWidth);
		for (auto& ch: input)
			ch = rand()%2;
		
		std::vector<cl_char> output(regionWidth);
		
		region.write(input, output);
		
		for (auto& ch: input)
			std::cout << " #"[ch];
		std::cout << "\n";
		for (auto& ch: output)
			std::cout << " #"[ch];
		std::cout << "\n\n" << std::flush;
		
	}
}

