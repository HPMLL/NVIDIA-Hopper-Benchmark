#!/bin/bash

if [ "$GPUArch" == "80" ] || [ "$GPUArch" == "89" ] || [ "$GPUArch" == "90" ]; then
	SM="$GPUArch"
elif [ "$GPUArch" == "" ]; then
    SM="90"
    echo "GPUArch isn't configured, use the default value 90"

else
	echo "GPUArch should be one of 80,89,90"
    exit
fi

if [ "$SILP" == "" ]; then
    echo "ILP for single_SM(SILP) isn't configured, use the default value 1"
	SILP="1"
fi
if [ "$AILP" == "" ]; then
    echo "ILP for all SM isn't configured, use the default value 4"
	AILP="4"
fi

echo "Compiler config: GPUArch=$SM, SILP=$SILP AILP=$AILP"

nvcc run_m16n8k128_single_SM.cu -gencode=arch=compute_$SM,code=sm_$SM -o single_SM  -DILP=$SILP
nvcc run_m16n8k128_all_SM.cu -gencode=arch=compute_$SM,code=sm_$SM -o all_SM -DILP=$AILP
