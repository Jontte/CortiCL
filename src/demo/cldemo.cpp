#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include <iostream>
#include <sstream>

bool spaceDown = false;
void checkEvents(bool* quit)
{
	SDL_Event e;
	while (SDL_PollEvent(&e))
	{
		if (e.type == SDL_QUIT)
			*quit = true;
		if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_SPACE)
		{
			spaceDown = true;
		}
		if (e.type == SDL_KEYUP && e.key.keysym.sym == SDLK_SPACE)
		{
			spaceDown = false;
		}
// 		if (e.type == SDL_MOUSEBUTTONDOWN)
// 			*quit = true;
	}
}

void demo1Loop(SDL_Window* window, bool& spaceDown);
void demo2Loop(SDL_Window* window, bool& spaceDown);
void demo3Loop(SDL_Window* window, bool& spaceDown);

int main(int argc, char *argv[])
{
	if (SDL_Init(SDL_INIT_EVERYTHING) == -1)
	{
		std::cout << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_Window* window = SDL_CreateWindow("cldemo", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 800, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
// 	SDL_Window* window = SDL_CreateWindow("cldemo", 200, 200, 800, 800, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	if (window == nullptr)
	{
		std::cout << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (renderer == nullptr)
	{
		std::cout << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_GLContext glContext = SDL_GL_CreateContext(window);
	SDL_GL_SetSwapInterval(0);

	int selection = 0;
	if (argc>1)
	{
		std::stringstream str;
		str << argv[1];
		str >> selection;
	}

	while(std::cin)
	{
		if (selection == 0)
		{
			std::cout << "Pick demo: \n\t1) stochastic pattern sweep\n\t2) periodic function prediction\n\t3) shape movement prediction\n> ";
			std::cin >> selection;
		}
		if (selection == 1)
		{
			demo1Loop(window, spaceDown);
			break;
		}
		else if(selection == 2)
		{
			demo2Loop(window, spaceDown);
			break;
		}
		else if(selection == 3)
		{
			demo3Loop(window, spaceDown);
			break;
		}
		selection = 0;
	}

	SDL_GL_DeleteContext(glContext);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
