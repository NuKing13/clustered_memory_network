#!/bin/bash

echo $( < counter )
$COUNTER < counter
rm counter
touch counter
COUNTER=$(($COUNTER + 1))
$COUNTER > counter

