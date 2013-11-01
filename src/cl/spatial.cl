
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

uint random(uint2* seedValue)
{
	uint seed = (seedValue->x++) + get_global_id(0);
	uint t = seed ^ (seed << 11);
	return seedValue->y ^ (seedValue->y >> 19) ^ (t ^ (t >> 8));
}
float randfloat(uint2* seedValue)
{
	uint i = random(seedValue) % 1000;
	return i / 1000.0;
}

void resetSynapse(global Synapse* synapse, int columnIndex, uint2* randomState)
{
	// Calculate a pseudorandom permanence value centered at CONNECTED_PERMANENCE
	float permanence = 0;
	permanence += randfloat(randomState);
	permanence -= randfloat(randomState);
	permanence *= 0.5;
	permanence += CONNECTED_PERMANENCE;
	if (permanence < 0.0)
		permanence = 0.0;
	if (permanence > 1.0)
		permanence = 1.0;
	synapse -> permanence = permanence;
	
	// Calculate pseudorandom target bit based on receptive field radius
	int columnX = columnIndex % REGION_WIDTH;
	int columnY = columnIndex / REGION_WIDTH;

	// Map column location in region to input space
	int iX = INPUT_WIDTH  * ((float)columnX) / REGION_WIDTH;
	int iY = INPUT_HEIGHT * ((float)columnY) / REGION_HEIGHT;
	
	int minX = 0; 
	int minY = 0;
	int maxX = INPUT_WIDTH;
	int maxY = INPUT_HEIGHT;
	
	if (RECEPTIVE_FIELD_RADIUS >= 0)
	{
		minX = max(0, iX - RECEPTIVE_FIELD_RADIUS);
		minY = max(0, iY - RECEPTIVE_FIELD_RADIUS);
		maxX = min(INPUT_WIDTH,  iX + RECEPTIVE_FIELD_RADIUS);
		maxY = min(INPUT_HEIGHT, iY + RECEPTIVE_FIELD_RADIUS);
	}

	int x = minX + random(randomState) % (maxX-minX);
	int y = minY + random(randomState) % (maxY-minY);
	synapse -> target = x + y * INPUT_WIDTH;
}

void kernel initRegion(
	global Column* columns,
	global Synapse* synapses,
	uint2 randomState)
{
	int columnIndex = get_global_id(0);
	global Column* column = &columns[columnIndex];
	uint2 state = randomState;
	
	// Column startup parameters
	column -> boost = 0.0;
	column -> overlap = 0.0;
	column -> active = false;
	column -> activeDutyCycle = 0.1;
	column -> minDutyCycle = 0.1;
	column -> overlapDutyCycle = 0.1;
	
	int synapseOffset = columnIndex * COLUMN_PROXIMAL_SYNAPSE_COUNT;
	for (int i = 0; i < COLUMN_PROXIMAL_SYNAPSE_COUNT; ++i)
	{
		global Synapse* synapse = &synapses[i + synapseOffset];	
		resetSynapse(synapse, columnIndex, &state);
	}
}

// Reset the worst synapse of each column
void kernel refineRegion(
	global Column* columns,
	global Synapse* synapses,
	uint2 randomState)
{
	int columnIndex = get_global_id(0);
	global Column* column = &columns[columnIndex];
	uint2 state = randomState;
		
	int synapseOffset = columnIndex * COLUMN_PROXIMAL_SYNAPSE_COUNT;
	int worstSynapseIndex = 0;
	float worstSynapsePermanence = 0;
	for (int i = 0; i < COLUMN_PROXIMAL_SYNAPSE_COUNT; ++i)
	{
		global Synapse* synapse = &synapses[i + synapseOffset];	
		if (i == 0 || synapse->permanence < worstSynapsePermanence)
		{
			worstSynapsePermanence = synapse->permanence;
			worstSynapseIndex = i;
		}
	}
	resetSynapse(&synapses[worstSynapseIndex + synapseOffset], columnIndex, &state);
}

void kernel computeOverlap(
	global Column* columns,
	global Synapse* synapses,
	global const char* input)
{
	int columnIndex = get_global_id(0);

	global Column* col = &columns[columnIndex];

	// Calculate the number of synapses that point to active input bits
	float overlap = 0;

	int synapseOffset = columnIndex * COLUMN_PROXIMAL_SYNAPSE_COUNT;
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
	global Synapse* synapses)
{
	int columnIndex = get_global_id(0);
	global Column* col = &columns[columnIndex];

	if (!col->active)
		return;

	// Given neighbourhood of nWidth*nHeight and total region topology of REGION_WIDTH*REGION_HEIGHT,
	// inhibit current column so that the neighbourhood has approximately SPARSITY_TARGET ratio of columns active

	int nWidth = INHIBITION_RADIUS;
	int nHeight = INHIBITION_RADIUS;
		
	int colX = columnIndex % REGION_WIDTH;
	int colY = columnIndex / REGION_WIDTH;

	int minX = colX-nWidth/2;
	int maxX = colX+nWidth/2+1;
	int minY = colY-nHeight/2;
	int maxY = colY+nHeight/2+1;

	if (nWidth == -1 || nHeight == -1)
	{
		// Global inhibition
		minX = 0;
		maxX = REGION_WIDTH;
		minY = 0;
		maxY = REGION_HEIGHT;
	}
	else
	{
		if (minX < 0) minX = 0;
		if (maxX > REGION_WIDTH) maxX = REGION_WIDTH;
		if (minY < 0) minY = 0;
		if (maxY > REGION_HEIGHT) maxY = REGION_HEIGHT;
	}

	int numActiveColumns = 0;

	int neighbours = (maxX-minX+1)*(maxY-minY+1);
	int n = SPARSITY_TARGET * neighbours;

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

				global Column* curColumn = &columns[y * REGION_WIDTH + x];

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

void kernel updatePermanences(
	global Column* columns,
	global Synapse* synapses,
	global const char* input)
{
	int columnIndex = get_global_id(0);

	global Column* col = &columns[columnIndex];
	int columnSynapseOffset = columnIndex * COLUMN_PROXIMAL_SYNAPSE_COUNT;

	if (col->active)
	{
		// Update permanences
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
	col->minDutyCycle = 0.01 * 0.1; // 0.1 = maxDutyCycle of neighbourhood
	col->activeDutyCycle =
		col->activeDutyCycle * DUTY_CYCLE_PERSISTENCE
		+ col->active * (1.0 - DUTY_CYCLE_PERSISTENCE);

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
