#include <stdexcept>
#include "../clregion.h"

int main()
{
	// Create OpenCL context
	CLContext context;
	// Create region

	int inputWidth = 80;
	int inputHeight = 1;

	int regionWidth = 80;
	int regionHeight = 1;

	int inhibitionRadius = 5;
	int receptiveFieldRadius = 5;

	CLArgs args;
	args.ColumnProximalSynapseCount = 5;
	args.ColumnProximalSynapseMinOverlap = 3;

	CLRegion region(context,
		CLTopology::localInhibition2D(inputWidth, inputHeight, regionWidth, regionHeight, inhibitionRadius, receptiveFieldRadius),
		args
	);

	int counter = 0;
	while (true)
	{
		counter++;

		// Create random input
		std::vector<cl_char> input(inputWidth);
		for (int i = 0; i < int(input.size()); ++i)
		{
			cl_char& ch = input[i];
			ch = rand()%2;
			if (counter % 10000 < inputWidth*50)
			{
				ch = abs(i-(counter%10000)/50) < (inputWidth / 16);
			}
		}

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

