#!/bin/bash
#input: n mod c r_stop

cd ../output/jureca_scans/

#transfer results
N=$1
MOD=$2
C=$3

if [ $N -eq 2500 ]
then
	R=7.0
elif [ $N -eq 5000 ]
then
	R=6.0
elif [ $N -eq 10000 ]
then
	R=5.0
else
	echo unecpected neuron number
	exit
fi

SRC="ostendorf1@jureca.fz-juelich.de:/p/project/jinm60/users/ostendorf1/Thesis_Git/clustered_memory_network/output/*mod${MOD}_c${C}*/res"
DEST="/home/noah/Thesis_Git/clustered_memory_network/output/jureca_scans/scan_n${N}_r${R}-${4}_step1.0_g5.0-32.0_step1.0_mod${MOD}_c${C}"

rsync -a $SRC $DEST

#clean up overscanned data

#CD="scan_n${N}_r${R}-32.0_step1.0_g5.0-32.0_step1.0_mod${MOD}_c${C}"

cd "${DEST}/res"


EXPORT="n${N}_r33-34_g33-36_mod${MOD}_c${C}"

mkdir tmp

cp $(ls | grep r33) tmp
cp $(ls | grep r34) tmp
cp $(ls | grep g33) tmp
cp $(ls | grep g34) tmp
cp $(ls | grep g35) tmp
cp $(ls | grep g36) tmp

rm $(ls | grep r33)
rm $(ls | grep r34)
rm $(ls | grep g33)
rm $(ls | grep g34)
rm $(ls | grep g35)
rm $(ls | grep g36)

mv tmp /home/noah/Thesis_Git/clustered_memory_network/output/jureca_scans/overscanned_data/$EXPORT

cd ../..

