#!/usr/bin/env bash

set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Process a directory with tweet files using MOA's Hoeffding tree

scriptName=`basename "$0"`
usage="Usage: $scriptName input_dir header_file output_file"
if [ $# -eq 0 ] ; then
  echo $usage
  exit 1
fi

# Define constants
JAR_FILE="../aggressionMOAClassification/target/aggression-moa-classification-1.0.jar"
MEMORY=512m

splitconf="0.01"
tiethre="0.05"
grace="200"
splitcrit="InfoGainSplitCriterion"  #"GiniSplitCriterion"

normalization="2"
preprocess="-p"
adaptive="-a"

frequency="1000"

# Process input parameters
data_dir=$1
header_file=$2
output_file=$3

# Process the tweets
java -Xmx${MEMORY} \
     -cp "${JAR_FILE}" \
     "run.ExecuteMoa" \
     " task.EvaluatePrequential" \
     " -s (tweets.TweetInstanceStream -f ${data_dir} -h ${header_file} -n ${normalization} ${preprocess} ${adaptive})" \
     " -l (trees.HoeffdingTree -c ${splitconf} -t ${tiethre} -g ${grace} -s ${splitcrit})" \
     " -e (BasicClassificationPerformanceEvaluator -o -p -r -f)" \
     " -f ${frequency} -i -1" 1> "${output_file}.out" 2> "${output_file}.err"

echo "Done"

