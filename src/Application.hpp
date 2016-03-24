#ifndef _Application
#define _Application

#include "includes/ezOptionParser.hpp"

using namespace ez;

const int REQUIRED_OPTION = 1;
const int NOT_REQUIRED_OPTION = 0;

int main(int argc, const char **argv);

void setupEzOptionParser(ezOptionParser& optionParser);

void showEzOptionParserUsage(ezOptionParser& optionParser);

#endif