#!/bin/bash
#Written by Matthew Golden to recompile a specific part of iS.e for testing 
#meant to be executed from src

cd obj
rm main.o
rm emissionfunction.o
rm arsenal.o
rm ParameterReader.o
rm readindata.o
rm Table.o
cd ..
rm iS.e
make
cd ..
rm iS.e
make
echo "All worked fine"
