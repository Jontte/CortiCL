#include <sstream>
#include "cltopology.h"

std::string CLTopology::serialize() const
{
	std::stringstream constants; constants
	<< "constant int INPUT_WIDTH = "            << inputWidth           << ";"
	<< "constant int INPUT_HEIGHT = "           << inputHeight          << ";"
	<< "constant int REGION_WIDTH = "           << regionWidth          << ";"
	<< "constant int REGION_HEIGHT = "          << regionHeight         << ";"
	<< "constant int INHIBITION_RADIUS = "      << inhibitionRadius     << ";"
	<< "constant int RECEPTIVE_FIELD_RADIUS = " << receptiveFieldRadius << ";";
	return constants.str();
}
