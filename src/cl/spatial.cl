
typedef struct
{
	float permanence;
	int target;
} Synapse;

typedef struct
{
	float boost;
	float overlap;
	bool active;

	float activeDutyCycle;
	float minDutyCycle;
	float overlapDutyCycle;
} Column;

void kernel computeOverlap(global Column* columns, global Synapse* synapses, global const char* input, int inputSize)
{
	int index = get_global_id(0);

	global Column* col = &columns[index];

	// Calculate the number of synapses that point to active input bits
	float overlap = 0;

	int synapseOffset = index * COLUMN_PROXIMAL_SYNAPSE_COUNT;
	for (int i = 0; i < COLUMN_PROXIMAL_SYNAPSE_COUNT; ++i)
	{
		global Synapse* syn = &synapses[synapseOffset + i];
		overlap +=
			(syn->permanence > CONNECTED_PERMANENCE) && input[syn->target];
	}

	col->active = false;

	if (overlap > COLUMN_PROXIMAL_SYNAPSE_MIN_OVERLAP)
	{
		col->active = true;
		overlap *= col->boost;
	}
	else
	{
		overlap = 0;
	}
	col->overlap = overlap;
}

void kernel inhibitNeighbours(
	global Column* columns,
	global Synapse* synapses,
	float sparsityTarget,
	int nWidth,
	int nHeight,
	int regionWidth,
	int regionHeight)
{
	int index = get_global_id(0);
	global Column* col = &columns[index];

	if (!col->active)
		return;

	// Given neighbourhood of nWidth*nHeight and total region topology of regionWidth*regionHeight,
	// inhibit current column so that the neighbourhood has approximately sparsityTarget ratio of columns active

	int colX = index % regionWidth;
	int colY = index / regionWidth;

	int minX = colX-nWidth/2;
	int maxX = colX+nWidth/2+1;
	int minY = colY-nHeight/2;
	int maxY = colY+nHeight/2+1;

	if (nWidth == -1 || nHeight == -1)
	{
		// Global inhibition
		minX = 0;
		maxX = regionWidth;
		minY = 0;
		maxY = regionHeight;
	}
	else
	{
		if (minX < 0) minX = 0;
		if (maxX > regionWidth) maxX = regionWidth;
		if (minY < 0) minY = 0;
		if (maxY > regionHeight) maxY = regionHeight;
	}

	int numActiveColumns = 0;

	int neighbours = (maxX-minX+1)*(maxY-minY+1);
	int n = sparsityTarget * neighbours;

	// Use partial selection sort to find the Nth activation
	float activationSkip = -1;
	for (int k = 0; k < n+1; k++)
	{
		float bestActivation = -1;
		int bestActCount = 0;

		for (int y = minY; y < maxY; ++y)
		{
			for (int x = minX; x < maxX; ++x)
			{
				if (x == colX && y == colY) continue;

				global Column* curColumn = &columns[y * regionWidth + x];

				float act = curColumn->overlap;

				if (activationSkip < 0 || act < activationSkip)
				{
					if (act == bestActivation)
					{
						bestActCount ++;
					}
					else if (act > bestActivation)
					{
						bestActivation = act;
						bestActCount = 0;
					}
				}
			}
		}
		k += bestActCount;
		activationSkip = bestActivation;
	}

	col->active = col->overlap >= activationSkip;
}


void kernel updatePermanences(global Column* columns, global Synapse* synapses, global const char* input)
{
	int index = get_global_id(0);

	global Column* col = &columns[index];
	int columnSynapseOffset = index * COLUMN_PROXIMAL_SYNAPSE_COUNT;
	// Update permanences
	if (col->active)
	{
		for (int i = 0; i < COLUMN_PROXIMAL_SYNAPSE_COUNT; ++i)
		{
			global Synapse* syn = &synapses[columnSynapseOffset + i];

			if (input[syn->target])
			{
				syn->permanence += PERMANENCE_STEP;
				if (syn->permanence > 1.0)
					syn->permanence = 1.0;
			}
			else
			{
				syn->permanence -= PERMANENCE_STEP;
				if (syn->permanence < 0.0)
					syn->permanence = 0.0;
			}
		}
	}

	// Update duty cycles

	col->minDutyCycle = 0.01 * 0.1; // maxDutyCycle of neighbourhood
	col->activeDutyCycle = col->activeDutyCycle * DUTY_CYCLE_PERSISTENCE + col->active * (1.0 - DUTY_CYCLE_PERSISTENCE);

	if (col->activeDutyCycle <= col->minDutyCycle)
		col->boost += BOOST_STEP;
	else
		col->boost = max(1.0, col->boost - BOOST_STEP);

	col->overlapDutyCycle = (col->overlapDutyCycle * DUTY_CYCLE_PERSISTENCE) + (col->activeDutyCycle > col->minDutyCycle) * (1.0 - DUTY_CYCLE_PERSISTENCE);

	if (col->overlapDutyCycle < col->minDutyCycle)
	{
		// Increase permanences
		for (int i = 0; i < COLUMN_PROXIMAL_SYNAPSE_COUNT; ++i)
		{
			global Synapse* syn = &synapses[columnSynapseOffset + i];

			syn->permanence += PERMANENCE_STEP;
			if (syn->permanence > 1.0)
				syn->permanence = 1.0;
		}
	}

}
