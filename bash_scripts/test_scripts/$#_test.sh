#!/bin/bash
if [ $# -eq 0 ]
then 
	echo 0 arguments
elif [ $# -eq 1 ]
then 
	echo 1 arguments
elif [ $# -eq 2 ]
then 
	echo 2 arguments
elif [ $# -eq 3 ]
then 
	echo 3 arguments
else
	echo more arguments
fi
