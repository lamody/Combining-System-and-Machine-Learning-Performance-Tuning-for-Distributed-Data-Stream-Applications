#!/usr/bin/env bash

set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Convert a set of multi-class arff files into binary class files
# Naming convention: tweets_p${prepro}_n${normal}_b0_nosp_f${numfeatures}.arff
#                 or tweets_p${prepro}_n${normal}_b0_a${adapt}_nosp_f${numfeatures}.arff

scriptName=`basename "$0"`
usage="Usage: $scriptName input_data_dir"
if [ $# -eq 0 ] ; then
  echo $usage
  exit 1
fi

# Declare constants #####################
declare -a prepros=("0" "1")
declare -a normals=("0" "1" "2" "3")
declare -a adapts=("0" "1") # Valid values: "" or "0" or "1"
numfeatures="16"
class="normal"

#########################################

# Find the python script
script=$(readlink -f "$0")
basedir=$(dirname "$script")
basedir=$(dirname "$basedir")
pythonScript="${basedir}/aggressionFeatureEvaluation/arff-convert-to-binary.py"

# Process input parameters
data_dir=$1
data_dir=${data_dir%/}

# Convert to binary
for prepro in "${prepros[@]}"
do
  for normal in "${normals[@]}"
  do
    for adapt in "${adapts[@]}"
    do
      if [ "$adapt" = "" ]
      then
        infile="${data_dir}/tweets_p${prepro}_n${normal}_b0_nosp_f${numfeatures}.arff"
        outfile="${data_dir}/tweets_p${prepro}_n${normal}_b1_nosp_f${numfeatures}.arff"
      else
        infile="${data_dir}/tweets_p${prepro}_n${normal}_b0_a${adapt}_nosp_f${numfeatures}.arff"
        outfile="${data_dir}/tweets_p${prepro}_n${normal}_b1_a${adapt}_nosp_f${numfeatures}.arff"
      fi

      echo $infile
      if [ -f "$infile" ]
      then
        python $pythonScript $infile $outfile $class
      fi

    done
  done
done

echo "Done!"

