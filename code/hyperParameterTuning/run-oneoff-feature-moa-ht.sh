#!/usr/bin/env bash

set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Remove one feature from the input data at a time
# and run MOA's Hoeffding tree

scriptName=`basename "$0"`
usage="Usage: $scriptName input_data_file output_dir"
if [ $# -eq 0 ] ; then
  echo $usage
  exit 1
fi

# Define constants
REMOVE_FEATURES_PY="/home/hero/projects/streaming-aggression/code/aggressionFeatureEvaluation/arff-remove-features.py"
MOA_DIR="/home/hero/deploy/moa-release-2019.05.0"
MEMORY=512m

splitconf="0.01" #"0.1"
tiethre="0.05" #"0.1"
grace="200"
splitcrit="InfoGainSplitCriterion"  #"GiniSplitCriterion"
binsplit="" #"-b"
nopreprune="" #"-p"
frequency="1000"
width="1000"

# Process input parameters
data_file=$1
output_dir=$2

# Create output directory
filename="$(basename ${data_file})"
filename="${filename%.*}"
output_dir=${output_dir%/}
output_dir="${output_dir}/${filename}"

mkdir -p ${output_dir}

# Get the features from the file
features=()
IFS=$'\n'
lines=(`head -n 40 ${data_file} | grep -i "@attribute"`)

for line in "${lines[@]}"
do
  IFS=$' '
  line_array=($line)
  features+=( "${line_array[1]}" )
done

unset 'features[${#features[@]}-1]'
#declare -p features

# Remove all features, one at a time, and run moa
for feature in "${features[@]}"
do
  out_filename="moa_ht_c${splitconf}_t${tiethre}_g${grace}_s${splitcrit}_b${binsplit}_p${nopreprune}_no${feature}"
  echo ${out_filename}

  python ${REMOVE_FEATURES_PY} "${data_file}" "${data_file}_no${feature}.arff" ${feature}

  java -Xmx${MEMORY} \
       -cp "${MOA_DIR}/lib/moa.jar:${MOA_DIR}/lib/*" \
       -javaagent:${MOA_DIR}/lib/sizeofag-1.0.4.jar moa.DoTask \
       "EvaluatePrequential -l (trees.HoeffdingTree -c ${splitconf} -t ${tiethre} -g ${grace} -s ${splitcrit} ${binsplit} ${nopreprune})" \
       " -s (ArffFileStream -f ${data_file}_no${feature}.arff)" \
       " -e (BasicClassificationPerformanceEvaluator -o -p -r -f)" \
       " -i 86000 -f ${frequency} -w ${width}" 1> "${output_dir}/${out_filename}.csv" 2> "${output_dir}/${out_filename}.log"

  rm "${data_file}_no${feature}.arff"

done


