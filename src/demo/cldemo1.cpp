#include <random>
#include "../clregion.h"
#include "util.h"
#include "sensor.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

void checkEvents(bool* quit);
void genData(int width, std::vector<cl_char>& data, double timer)
{
	static std::mt19937 gen;

	for (int a = 0; a < width; ++a)
	{
		for (int i = 0; i < width; ++i)
		{
			double x = double(i)/width - 0.5;
			double y = double(a)/width - 0.5;
			double p = pow(sin(x+y+timer*0.01), 20);

			bool k = std::bernoulli_distribution(p)(gen);

			data[i+width*a] = k;
		}
	}
}

void demo1Loop(SDL_Window* window, bool& spaceDown)
{
	// init opencl context here
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

	int columnWidth = 32;
	int columns = columnWidth*columnWidth;
	int inputWidth = 32;
	int inputSize = inputWidth*inputWidth;

	int inhibitionRadius = 5;
	int receptiveFieldRadius = 5;
	CLRegion region(defaultDevice, context,
					CLTopology::localInhibition2D(inputWidth, inputWidth, columnWidth, columnWidth, inhibitionRadius, receptiveFieldRadius),
					CLArgs()
	);

	double timer = 0;

	int lastRPS = 0;
	int curRPS = 0;
	time_t lastTime = time(nullptr);

	glTranslatef(-0.5, -0.5, 0);

	std::vector<cl_char> dataIn(inputSize);
	std::vector<cl_char> dataOut(columns);
	while(!done)
	{
		curRPS ++;
		time_t timeNow = time(nullptr);
		if (timeNow != lastTime)
		{
			lastRPS = curRPS;
			curRPS = 0;
			lastTime = timeNow;

			CLStats stats = region.getStats();
			std::cout << lastRPS << " " << stats.averageBoost << " " << stats.averageDutyCycle << std::endl;
		}

		checkEvents(&done);

		SDL_GL_SwapWindow(window);
		glClearColor ( 0.0, 0.0, 0.0, 1.0 );
		glClear ( GL_COLOR_BUFFER_BIT );
		timer += 1;

		// Generate some data

		genData(inputWidth, dataIn, timer);

		glColor3d(1,1,1);

		glBegin(GL_QUADS);
		for (int a = 0; a < inputWidth; ++a)
		{
			for (int i = 0; i < inputWidth; ++i)
			{
				double x = double(i)/(inputWidth-1) - 0.5;
				double y = double(a)/(inputWidth-1) - 0.5;
				double nx = double(i+1)/(inputWidth-1) - 0.5;
				double ny = double(a+1)/(inputWidth-1) - 0.5;

				bool k = dataIn[i+inputWidth*a];
				if (k)
				{
					glVertex2d(x, y);
					glVertex2d(nx, y);
					glVertex2d(nx, ny);
					glVertex2d(x, ny);
				}
			}
		}
		glEnd();
		region.write(dataIn, dataOut);

		glBegin(GL_QUADS);
		for (int a = 0; a < columnWidth; ++a)
		{
			for (int i = 0; i < columnWidth; ++i)
			{
				double x = double(i)/(columnWidth-1) - 0.5;
				double y = double(a)/(columnWidth-1) - 0.5;
				double nx = double(i+1)/(columnWidth-1) - 0.5;
				double ny = double(a+1)/(columnWidth-1) - 0.5;

				bool k = dataOut[i+columnWidth*a];
				if (k)
				{
					glVertex2d(1+x, y);
					glVertex2d(1+nx, y);
					glVertex2d(1+nx, ny);
					glVertex2d(1+x, ny);
				}
			}
		}
		glEnd();
		std::vector<double> cast;
		region.backwards(dataOut, cast);
		glBegin(GL_QUADS);
		for (int a = 0; a < inputWidth; ++a)
		{
			for (int i = 0; i < inputWidth; ++i)
			{
				double x = double(i)/(inputWidth-1) - 0.5;
				double y = double(a)/(inputWidth-1) + 0.5;
				double nx = double(i+1)/(inputWidth-1) - 0.5;
				double ny = double(a+1)/(inputWidth-1) + 0.5;

				double k = cast[i+inputWidth*a];
				k /= 5;
				if (k > 1) k = 1;
				glColor3d(k, k, k);

				glVertex2d(x, y);
				glVertex2d(nx, y);
				glVertex2d(nx, ny);
				glVertex2d(x, ny);
			}
		}
		glEnd();
	}
}
