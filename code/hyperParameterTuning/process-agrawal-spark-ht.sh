#!/usr/bin/env bash

# set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Process a directory with tweet files using Spark's Hoeffding tree
# Example usage with local dir:
# ./process-tweets-spark.sh file:///home/hero/projects/streaming-aggression/data/mix_b1/ ../../data/100k_tweets/arff/labeled_tweets_prepro_bin.arff.head output
#
# Example usage with hdfs:
# ./process-tweets-spark.sh hdfs://dicl15.cut.ac.cy:9000/user/hadoop/inputbin ../../data/100k_tweets/arff/tweets_p1_n2_b1_a0_nosp.arff.head output

scriptName=`basename "$0"`
script=$(readlink -f "$0")
basedir=$(dirname $(dirname "$script") )
usage="Usage: $scriptName input_dir header_file output_file_prefix"
if [ $# -eq 0 ] ; then
  echo $usage
  exit 1
fi

# Define constants
SPARK_HOME="/home1/hadoop/deploy/spark"
SPARK_MASTER="spark://dicl15.cut.ac.cy:7077" #  "local[*]"
# SPARK_HOME="/opt/spark"
# SPARK_MASTER="spark://localhost:7077" #  "local[*]"
JAR_FILE="${basedir}/aggressionStreamClassification/target/scala-2.11/aggression-stream-classification-assembly-0.1.jar"

batchInterval=1000

maxReceivers=4  # must be less than number of threads in Spark
maxCounts=( 100 200 300 400 200 200 200 100 400 )
rateLimits=( 16 16 16 16 8 24 32 48 12 )

# declare -a splitconfs=("0.01" "0.1") # "0.5")
# declare -a tiethres=("0.05" "0.1")
# declare -a graces=("200" "500")
# declare -a splitcrits=("InfoGainSplitCriterion" "GiniSplitCriterion")
# declare -a depths=("20" "30")

declare -a splitconfs=("0.001" "0.01" "0.1")
declare -a tiethres=("0.001" "0.05" "0.1")
declare -a graces=("50" "800" "1500")
declare -a splitcrits=("InfoGainSplitCriterion" "GiniSplitCriterion")
declare -a depths=("20" "30")

normalization="2"

# Process input parameters
data_dir=$1
header_file=$2
output_dir=$3

mkdir -p ${output_dir}

# Process the tweets
for idx in "${!maxCounts[@]}"; do
  maxCount=${maxCounts[$idx]}
  rateLimit=${rateLimits[$idx]}
  s="s$((idx+1))"
  out_dir="${output_dir}/${s}"
  mkdir -p ${out_dir}

  iter=1
  total=$((${#splitconfs[@]} * ${#tiethres[@]} * ${#graces[@]} * ${#splitcrits[@]} * ${#depths[@]}))

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
            out_filename="spark_ht_c${splitconf}_t${tiethre}_g${grace}_s${splitcrit}_d${depth}"
            echo "${s} - ${out_filename} ($iter/$total)"

            $SPARK_HOME/bin/spark-submit \
              --class "aggression.aggressionStreamJob" \
              --master "${SPARK_MASTER}" \
              "${JAR_FILE}" \
              "${batchInterval}" \
              " aggression.tasks.EvaluateAgrawal" \
              " -t (aggression.arff.ArffDirReader -p ${data_dir} -c ${maxCount} -r ${maxReceivers} -l ${rateLimit})" \
              " -f (aggression.arff.FeatureExtractor -f ${header_file} -n ${normalization})" \
              " -l (org.apache.spark.streamdm.classifiers.trees.HoeffdingTree -c ${splitconf} -t ${tiethre} -g ${grace} -s ${splitcrit} -h ${depth})" \
              " -h" 1> "${out_dir}/${out_filename}.csv" 2> "${out_dir}/${out_filename}.log"
          
            iter=$((iter+1))

            # latest=$(ls /opt/spark/logs/ -t | head -n 1)
            # mv "/opt/spark/logs/${latest}" "${out_dir}/${out_filename}"

            latest=$(/home1/hadoop/deploy/hadoop/bin/hdfs dfs -ls -t -C spark-events | grep "app*" | head -n 1)
            /home1/hadoop/deploy/hadoop/bin/hdfs dfs -copyToLocal ${latest} "${out_dir}/${out_filename}"
          done
        done
      done
    done
  done
done

echo "Done"
