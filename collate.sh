#!/bin/bash

cat << EOF
#pragma once
#include <array>
#include <numeric>
#include <string>
EOF


for name in $@; do
	printf "\n\n\n\n"
	printf "// ============================================================================\n"
	cat $name\
	| awk '/ND_API_START/{a=1}/ND_API_END/{print;a=0}a'\
	| sed 's/\/\/ ND_API_START//'\
	| sed 's/\/\/ ND_API_END//'
done


for name in $@; do
	printf "\n\n\n\n"
	printf "// ============================================================================\n"
	cat $name\
	| awk '/ND_IMPL_START/{a=1}/ND_IMPL_END/{print;a=0}a'\
	| sed 's/\/\/ ND_IMPL_START//'\
	| sed 's/\/\/ ND_IMPL_END//'
done
