#!/usr/bin/env bash

set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Perform hyper-parameter tuning for Spark's Multi Class classifier

scriptName=`basename "$0"`
usage="Usage: $scriptName input_arff_file output_dir"
if [ $# -eq 0 ] ; then
  echo $usage
  exit 1
fi

# Define constants ##############################
batchInterval=200  # in ms
chunkSize=1000
instanceLimit=86000
slideDuration=200  # in ms

declare -a splitconfs=("0.01" "0.1")
declare -a tiethres=("0.05" "0.1")
declare -a graces=("200" "500")
declare -a splitcrits=("InfoGainSplitCriterion") # "GiniSplitCriterion")
declare -a depths=("20" "30")

#################################################
# Get input parameters
data_file=$1
output_dir=$2

# Create output directory
filename="$(basename ${data_file})"
filename="${filename%.*}"
output_dir=${output_dir%/}
output_dir="${output_dir}/${filename}"

mkdir -p ${output_dir}

# Find the spark.sh script
script=$(readlink -f "$0")
basedir=$(dirname "$script")
basedir=$(dirname "$basedir")
run_spark=${basedir}/aggressionStreamClassification/spark.sh

# Perform tuning using grid-based approach
for splitconf in "${splitconfs[@]}"
do
 for tiethre in "${tiethres[@]}"
 do
  for grace in "${graces[@]}"
  do
   for splitcrit in "${splitcrits[@]}"
   do
    for depth in "${depths[@]}"
    do
      out_filename="spark_mc_c${splitconf}_t${tiethre}_g${grace}_s${splitcrit}_d${depth}"
      echo ${out_filename}

      ${run_spark} "${batchInterval} EvaluatePrequential" \
        " -l (org.apache.spark.streamdm.classifiers.MultiClassLearner " \
        " -l (org.apache.spark.streamdm.classifiers.trees.HoeffdingTree -c ${splitconf} -t ${tiethre} -g ${grace} -h ${depth}))" \
        " -s (FileReader -k ${chunkSize} -i ${instanceLimit} -d ${slideDuration} -f ${data_file}) " \
        "-e (aggression.evaluation.ExtendedClassificationEvaluator) -h" \
        1> "${output_dir}/${out_filename}.csv" 2> "${output_dir}/${out_filename}.log"

    done
   done
  done
 done
done


