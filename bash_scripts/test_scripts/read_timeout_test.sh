#!/bin/bash

echo Input please
read -t 5 INPUT

if [ "${INPUT}" = "" ]
then
	echo no input success
	exit
else 
	echo "${INPUT}"
	exit
fi

echo ended
