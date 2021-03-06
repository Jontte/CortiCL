#include "clargs.h"
#include <sstream>

std::string CLArgs::serialize() const
{
	// Write constants to a single source line. This way any line numbers reported by the OpenCL compiler will still be valid.

	std::stringstream constants; constants
	<< "constant int COLUMN_PROXIMAL_SYNAPSE_COUNT = "       << ColumnProximalSynapseCount      << ";"
	<< "constant int COLUMN_PROXIMAL_SYNAPSE_MIN_OVERLAP = " << ColumnProximalSynapseMinOverlap << ";"
	<< "constant float BOOST_STEP = "                        << BoostStep                       << ";"
	<< "constant float DUTY_CYCLE_PERSISTENCE = "            << DutyCyclePersistence            << ";"
	<< "constant float SPARSITY_TARGET = "                   << SparsityTarget                  << ";"
	<< "constant int COLUMN_CELL_COUNT = "                   << ColumnCellCount                 << ";"
	<< "constant int CELL_SEGMENT_COUNT = "                  << CellSegmentCount                << ";"
	<< "constant int SEGMENT_SYNAPSE_COUNT = "               << SegmentSynapseCount             << ";"
	<< "constant int SEGMENT_ACTIVATION_THRESHOLD = "        << SegmentActivationThreshold      << ";"
	<< "constant int SEGMENT_MIN_THRESHOLD = "               << SegmentMinThreshold             << ";"
	<< "constant float CONNECTED_PERMANENCE = "              << ConnectedPermanence             << ";"
	<< "constant float PERMANENCE_STEP = "                   << PermanenceStep                  << ";";

	return constants.str();
}
