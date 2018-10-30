#!/bin/bash


cat selector.hpp\
| awk '/ND_NAMESPACE_START/{a=1}/ND_NAMESPACE_END/{print;a=0}a'\
| sed 's/\/\/ ND_NAMESPACE_START//'\
| sed 's/\/\/ ND_NAMESPACE_END//'\


cat selector.hpp\
| awk '/ND_CLASS_START/{a=1}/ND_CLASS_END/{print;a=0}a'\
| sed 's/\/\/ ND_CLASS_START//'\
| sed 's/\/\/ ND_CLASS_END//'\
