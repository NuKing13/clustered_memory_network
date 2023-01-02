#!/bin/bash

TEST=$(ssh test_squeue | grep ostendor)
if [ "${TEST}" != "" ]
then
	echo jobs queued
else 
	echo no jobs queued
fi
