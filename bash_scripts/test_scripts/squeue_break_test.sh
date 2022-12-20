#!/bin/bash

TEST=$(ssh test_squeue_grep | grep jrc0014)
if [ "${TEST}" = "" ]
then
	echo no jobs queued
else 
	echo jobs queued
fi
