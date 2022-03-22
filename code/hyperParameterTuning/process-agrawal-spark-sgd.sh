#!/usr/bin/env bash

# set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Process a directory with agrawal files using Spark's SGD
# Example usage with local dir:
# ./process-agrawal-spark-sgd.sh file:///home/hero/projects/streaming-aggression/data/mix_b1/ ../../data/100k_tweets/arff/labeled_tweets_prepro_bin.arff.head output
#
# Example usage with hdfs:
# ./process-agrawal-spark-sgd.sh hdfs://dicl15.cut.ac.cy:9000/user/hadoop/inputbin ../../data/100k_tweets/arff/tweets_p1_n2_b1_a0_nosp.arff.head output

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

# declare -a lambdas=("0.001" "0.01" "0.05" "0.1")
# declare -a lossFuns=("LogisticLoss" "SquaredLoss" "HingeLoss" "PerceptronLoss")
# declare -a regulars=("ZeroRegularizer" "L2Regularizer")
# declare -a regParams=("0.001" "0.01")

declare -a lambdas=("0.001" "0.01" "0.05" "0.1")
declare -a lossFuns=("LogisticLoss" "SquaredLoss" "HingeLoss" "PerceptronLoss")
declare -a regulars=("ZeroRegularizer" "L2Regularizer")
declare -a regParams=("0.001" "0.01")

normalization="2"

# Process input parameters
data_dir=$1
header_file=$2
output_dir=$3

mkdir -p ${output_dir}

# Calculate the number of features
features=`grep -i @attribute "${header_file}" | wc -l`
features="$(($features-1))"

# Process agrawal data
for idx in "${!maxCounts[@]}"
do
    maxCount=${maxCounts[$idx]}
    rateLimit=${rateLimits[$idx]}
    s="s$((idx+1))"
    out_dir="${output_dir}/${s}"
    mkdir -p ${out_dir}

    iter=1
    total=$((${#lambdas[@]} * ${#lossFuns[@]} * ${#regulars[@]} * ${#regParams[@]}))

    for lambda in "${lambdas[@]}"
    do
        for lossFun in "${lossFuns[@]}"
        do
            for regular in "${regulars[@]}"
            do
                for regParam in "${regParams[@]}"
                do
                    out_filename="spark_sgd_l${lambda}_o${lossFun}_r${regular}_p${regParam}"
                    echo "${s} - ${out_filename} ($iter/$total)"

                    $SPARK_HOME/bin/spark-submit \
                    --class "aggression.aggressionStreamJob" \
                    --master "${SPARK_MASTER}" \
                    "${JAR_FILE}" \
                    "${batchInterval}" \
                    " aggression.tasks.EvaluateAgrawal" \
                    " -t (aggression.arff.ArffDirReader -p ${data_dir} -c ${maxCount} -r ${maxReceivers} -l ${rateLimit})" \
                    " -f (aggression.arff.FeatureExtractor -f ${header_file} -n ${normalization})" \
                    " -l (org.apache.spark.streamdm.classifiers.SGDLearner -f ${features} -l ${lambda} -o ${lossFun} -r ${regular} -p ${regParam})" \
                    " -e (aggression.evaluation.ExtendedClassificationEvaluator)" \
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

echo "Done"
