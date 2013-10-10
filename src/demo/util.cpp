#include "util.h"

#include <stdio.h>

std::string set_color(Color foreground, Color background)
{
	char num_s[3];
	std::string s = "\033[";

	if (foreground == Color::NONE && background == Color::NONE) s += "0"; // reset colors if no params

	if (foreground != Color::NONE)
	{
		sprintf(num_s, "%d", 29 + int(foreground));
		s += num_s;

		if (background != Color::NONE) s += ";";
	}

	if (background != Color::NONE) {
		sprintf(num_s, "%d", 39 + int(background));
		s += num_s;
	}

	return s + "m";
}