#!/usr/bin/env bash

set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Process a directory with tweet files using Spark's Hoeffding tree
# Example usage with local dir:
# ./process-tweets-spark.sh file:///home/hero/projects/streaming-aggression/data/mix_b1/ ../../data/100k_tweets/arff/labeled_tweets_prepro_bin.arff.head output
#
# Example usage with hdfs:
# ./process-tweets-spark.sh hdfs://localhost:9000/user/hadoop/input ../../data/100k_tweets/arff/tweets_p1_n0_b0_a1_nosp_f16.arff.head output


scriptName=`basename "$0"`
script=$(readlink -f "$0")
basedir=$(dirname $(dirname "$script") )
usage="Usage: $scriptName input_dir header_file output_file"
if [ $# -eq 0 ] ; then
  echo $usage
  exit 1
fi

# Define constants
SPARK_HOME="/home/hero/deploy/spark-2.3.2"
SPARK_MASTER="local[2]"
JAR_FILE="${basedir}/aggressionStreamClassification/target/scala-2.11/aggression-stream-classification-assembly-0.1.jar"

batchInterval="1000"

maxCount=1000
maxReceivers=1  # must be less than number of threads in Spark
rateLimit=0     # 0 = unlimited

splitconf="0.01"
tiethre="0.05"
grace="200"
splitcrit="InfoGainSplitCriterion"  #"GiniSplitCriterion"

normalization="2"
preprocess="-p"
adaptive="-a"

# Process input parameters
data_dir=$1
header_file=$2
output_file=$3

# Process the tweets
$SPARK_HOME/bin/spark-submit \
    --class "aggression.aggressionStreamJob" \
    --master "${SPARK_MASTER}" \
    "${JAR_FILE}" \
    "${batchInterval}" \
    " aggression.tasks.EvaluateTweetAggression" \
    " -t (aggression.tweet.TweetDirReader -p ${data_dir} -c ${maxCount} -r ${maxReceivers} -l ${rateLimit})" \
    " -f (aggression.tweet.FeatureExtractor -f ${header_file} -n ${normalization} ${preprocess} ${adaptive})" \
    " -l (org.apache.spark.streamdm.classifiers.trees.HoeffdingTree -c ${splitconf} -t ${tiethre} -g ${grace} -s ${splitcrit})" \
    " -h" 1> "${output_file}.out" 2> "${output_file}.err"

echo "Done"

