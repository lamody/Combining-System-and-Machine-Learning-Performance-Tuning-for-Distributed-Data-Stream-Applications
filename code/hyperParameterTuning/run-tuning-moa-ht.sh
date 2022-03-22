#!/usr/bin/env bash

set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Perform hyper-parameter tuning for MOA's hoeffding tree

scriptName=`basename "$0"`
usage="Usage: $scriptName input_data_file output_dir"
if [ $# -eq 0 ] ; then
  echo $usage
  exit 1
fi

# Get input parameters
data_file=$1
output_dir=$2

# Create output directory
filename="$(basename ${data_file})"
filename="${filename%.*}"
output_dir=${output_dir%/}
output_dir="${output_dir}/${filename}"

mkdir -p ${output_dir}

# Define constants
MOA_DIR="/home/hero/deploy/moa-release-2019.05.0"
MEMORY=512m

declare -a splitconfs=("0.01" "0.1") #"0.5")
declare -a tiethres=("0.05" "0.1")
declare -a graces=("200") # "500)"
declare -a splitcrits=("InfoGainSplitCriterion" "GiniSplitCriterion")
declare -a binsplits=("") # "-b")
declare -a nopreprunes=("" "-p")
frequency="1000"
width="1000"

# Perform tuning using grid-based approach
for splitconf in "${splitconfs[@]}"
do
 for tiethre in "${tiethres[@]}"
 do
  for grace in "${graces[@]}"
  do
   for splitcrit in "${splitcrits[@]}"
   do
    for binsplit in "${binsplits[@]}"
    do
     for nopreprune in "${nopreprunes[@]}"
     do

      out_filename="moa_ht_c${splitconf}_t${tiethre}_g${grace}_s${splitcrit}_b${binsplit}_p${nopreprune}"
      echo ${out_filename}

      java -Xmx${MEMORY} \
       -cp "${MOA_DIR}/lib/moa.jar:${MOA_DIR}/lib/*" \
       -javaagent:${MOA_DIR}/lib/sizeofag-1.0.4.jar moa.DoTask \
       "EvaluatePrequential -l (trees.HoeffdingTree -c ${splitconf} -t ${tiethre} -g ${grace} -s ${splitcrit} ${binsplit} ${nopreprune})" \
       " -s (ArffFileStream -f ${data_file})" \
       " -e (BasicClassificationPerformanceEvaluator -o -p -r -f)" \
       " -i 86000 -f ${frequency} -w ${width}" 1> "${output_dir}/${out_filename}.csv" 2> "${output_dir}/${out_filename}.log"

     done
    done
   done
  done
 done
done


