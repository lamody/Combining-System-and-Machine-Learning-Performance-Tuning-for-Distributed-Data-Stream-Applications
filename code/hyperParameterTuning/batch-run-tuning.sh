#!/usr/bin/env bash

set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Execute the script 'run-tuning-*.sh' for each matched file in a directory
# Naming convention: tweets_p${prepro}_n${normal}_b${binary}_nosp_f${numfeatures}.arff
#                 or tweets_p${prepro}_n${normal}_b${binary}_a${adapt}_nosp_f${numfeatures}.arff

scriptName=`basename "$0"`
usage="Usage: $scriptName (moa|spark) (ht|sgd|mc|rf) input_data_dir output_data_dir"
if [ $# -eq 0 ] ; then
  echo $usage
  exit 1
fi

# Declare constants ########################
declare -a prepros=("0" "1")
declare -a normals=("0" "2")
declare -a binarys=("0" "1")
declare -a adapts=("0" "1") # Valid values: "" or "0" or "1"
numfeatures="16"

############################################

# Find the tuning script to run
script=$(readlink -f "$0")
basedir=$(dirname "$script")
tuning_script="run-tuning-$1-$2.sh"
tuning_script="${basedir}/${tuning_script}"

if [ ! -f "${tuning_script}" ]
then
  echo "ERROR: the script file does not exit: "
  echo "${tuning_script}"
  exit 1
fi

# Process input parameters
data_dir=$3
data_dir=${data_dir%/}

output_dir=$4
output_dir=${output_dir%/}

# Perform tuning
for prepro in "${prepros[@]}"
do
  for normal in "${normals[@]}"
  do
    for binary in "${binarys[@]}"
    do
      for adapt in "${adapts[@]}"
      do
        if [ "$adapt" = "" ]
        then
          infile="${data_dir}/tweets_p${prepro}_n${normal}_b${binary}_nosp_f${numfeatures}.arff"
        else
          infile="${data_dir}/tweets_p${prepro}_n${normal}_b${binary}_a${adapt}_nosp_f${numfeatures}.arff"
        fi

        echo $infile
        if [ -f "$infile" ]
        then
          ${tuning_script} ${infile} ${output_dir}
        fi

      done
    done
  done
done

echo "Done!"

