#include <thread>
#include "../clregion.h"
#include "util.h"
#include "sensor.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

void checkEvents(bool* quit);


static void drawCircle(double x, double y, double radius)
{
	glBegin(GL_TRIANGLE_FAN);
	glVertex2d(x, y);
	int verts = 40;
	for (int i = 0; i < verts+1; ++i)
	{
		double angle = 2*M_PI* double(i)/verts;
		glVertex2d(x + radius * cos(angle), y + radius * sin(angle));
	}
	glEnd();
}

template <class T>
void drawLine(double x1, double y1, double x2, double y2, double width, const T& bits)
{
	double len = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
	double step = len / bits.size();

	double dx = (x2-x1) / len;
	double dy = (y2-y1) / len;

	double tx = dy;
	double ty = -dx;

	glBegin(GL_QUADS);
	for (int i = 0; i < int(bits.size()); ++i)
	{
		double d = bits[i] / 10.0;
		glColor3d(d, d, d);
		glVertex2d(x1+i * dx * step     - tx * width/2, y1+i * dy * step     - ty * width/2);
		glVertex2d(x1+(i+1) * dx * step - tx * width/2, y1+(i+1) * dy * step - ty * width/2);
		glVertex2d(x1+(i+1) * dx * step + tx * width/2, y1+(i+1) * dy * step + ty * width/2);
		glVertex2d(x1+i * dx * step     + tx * width/2, y1+i * dy * step     + ty * width/2);
	}
	glEnd();
}

void demo3Loop(SDL_Window* window, bool& spaceDown)
{
	std::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);

	if (platformList.empty())
	{
		throw std::runtime_error("No OpenCL platforms available");
	}

	auto& defaultPlatform = platformList.front();
	std::vector< cl::Device > deviceList;
	defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &deviceList);
	if (deviceList.empty())
	{
		throw std::runtime_error("OpenCL platform contains no devices");
	}

	auto& defaultDevice = deviceList.front();
	cl::Context context({defaultDevice});

	bool done = false;
	glEnable( GL_BLEND );

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// Begin simulation init

	// We need two sensors, one for each axis
	int sensorResolution = 100;
	int sensorWindowSize = 10;
	Sensor sensorX(sensorResolution, sensorWindowSize);
	Sensor sensorY(sensorResolution, sensorWindowSize);

	// Input to the network is the two sensor readings concatenated
	int inputSize = sensorResolution * 2;
	int columns = 100;

	int inhibitionRadius = 5;
	int receptiveFieldRadius = 5;
	CLArgs args;
 	args.ColumnProximalSynapseMinOverlap = 3;
 	args.ColumnProximalSynapseCount = 10;

	CLRegion region(defaultDevice, context,
		CLTopology::line(inputSize, columns, inhibitionRadius, receptiveFieldRadius),
		args
	);

	std::vector<cl_char> dataIn(inputSize);
	std::vector<cl_char> dataOut(columns);

	std::vector<double> noisyRemap(inputSize);

	auto predict = [&](double inputX, double inputY)
	{
		// Encode to SDR via an instance of the Sensor class
		auto dataInX = sensorX.encode(inputX);
		auto dataInY = sensorY.encode(inputY);

		// Concatenate sensor readings to form input
		dataIn.clear();
		dataIn.insert(dataIn.end(), dataInX.begin(), dataInX.end());
		dataIn.insert(dataIn.end(), dataInY.begin(), dataInY.end());

		// Feed input to region, receive activation in dataOut
		region.write(dataIn, dataOut, false);

		// Find out what kind of input would cause this kind of region activation (noisy backwards convolution)
		region.backwards(dataOut, noisyRemap);

		// Use sensor to find out approximate input value that would cause this kind of SDR
		// First split reading to two
		std::vector<double> dataX(noisyRemap.begin(), noisyRemap.begin() + sensorResolution);
		std::vector<double> dataY(noisyRemap.begin() + sensorResolution, noisyRemap.end());

		drawLine(-1, -1, 1, -1, 0.05, dataInX);
		drawLine(-1, -1+0.1, 1, -1+0.1, 0.05, dataX);

		drawLine(-1, -1, -1, 1, 0.05, dataInY);
		drawLine(-1+0.1, -1, -1+0.1, 1, 0.05, dataY);

		return std::make_pair(sensorX.decode(dataX), sensorY.decode(dataY));
	};

	double timer = 0;
	while(!done)
	{
		timer += M_PI / 2.1;
		checkEvents(&done);

		SDL_GL_SwapWindow(window);
		glClearColor ( 0.0, 0.0, 0.0, 1.0 );
		glClear ( GL_COLOR_BUFFER_BIT );

		// Current circle coordinates...
		double x = cos(timer) / 2;
		double y = sin(timer) / 2;

		// Send to region
		std::pair<double, double> nextCoord = predict(x, y);

		// Draw current and next

		double radius = 0.25;
		glColor4d(1, 1, 1, 1);
		drawCircle(x, y, radius);
		glColor4d(1, 0, 0, 0.5);
		drawCircle(nextCoord.first, nextCoord.second, radius);

		if (!spaceDown)
			std::this_thread::sleep_for(std::chrono::milliseconds(1000/3));

		//CLStats stats = region.getStats();
		//std::cout << stats.averageBoost << " " << stats.averageDutyCycle << " " << stats.totalSegments << " " << stats.totalSynapses << std::endl;
	}
}
