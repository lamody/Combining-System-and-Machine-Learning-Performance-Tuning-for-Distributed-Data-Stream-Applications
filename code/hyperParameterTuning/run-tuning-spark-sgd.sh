#!/usr/bin/env bash

# set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Perform hyper-parameter tuning for Spark's SGD Learner

scriptName=`basename "$0"`
usage="Usage: $scriptName input_arff_file output_dir"
if [ $# -eq 0 ] ; then
  echo $usage
  exit 1
fi

# Define constants ##############################
batchInterval=1000  # in ms
# chunkSize=200
chunkSize=2000
# instanceLimit=1000
instanceLimit=50000
slideDuration=100  # in ms

declare -a lambdas=("0.001" "0.01" "0.05" "0.1")
declare -a lossFuns=("LogisticLoss" "SquaredLoss" "HingeLoss" "PerceptronLoss")
declare -a regulars=("ZeroRegularizer" "L2Regularizer")
declare -a regParams=("0.001" "0.01")

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

# Calculate the number of features
features=`grep -i @attribute "${data_file}" | wc -l`
features="$(($features-1))"

iter=1
total=$((${#lambdas[@]} * ${#lossFuns[@]} * ${#regulars[@]} * ${#regParams[@]}))

cp /opt/spark/conf/spark-defaults.conf ${output_dir}/spark-defaults.conf
cp /opt/spark/conf/spark-env.sh ${output_dir}/spark-env.sh

# cp /home1/hadoop/deploy/spark/conf/spark-defaults.conf ${output_dir}/spark-defaults.conf
# cp /home1/hadoop/deploy/spark/conf/spark-env.sh ${output_dir}/spark-env.sh

# Perform tuning using grid-based approach
for lambda in "${lambdas[@]}"
do
 for lossFun in "${lossFuns[@]}"
 do
  for regular in "${regulars[@]}"
  do
   for regParam in "${regParams[@]}"
   do
      out_filename="spark_sgd_l${lambda}_o${lossFun}_r${regular}_p${regParam}"
      echo "${out_filename} ($iter/$total)"

      ${run_spark} "${batchInterval} EvaluatePrequential" \
        " -l (org.apache.spark.streamdm.classifiers.SGDLearner -f ${features} -l ${lambda} -o ${lossFun} -r ${regular} -p ${regParam})" \
        " -s (FileReader -k ${chunkSize} -i ${instanceLimit} -d ${slideDuration} -f ${data_file})" \
        " -e (aggression.evaluation.ExtendedClassificationEvaluator) -h" \
        1> "${output_dir}/${out_filename}.csv" 2> "${output_dir}/${out_filename}.log"

      iter=$((iter+1))

      latest=$(ls /opt/spark/logs/ -t | head -n 1)
      cp "/opt/spark/logs/${latest}" "${output_dir}/${out_filename}"
      # ln -s "/opt/spark/logs/${latest}" "${output_dir}/${out_filename}"

      # latest=$(/home1/hadoop/deploy/hadoop/bin/hdfs dfs -ls -t -C spark-events | head -n 1)
      # /home1/hadoop/deploy/hadoop/bin/hdfs dfs -copyToLocal ${latest} "${output_dir}/${out_filename}"
   done
  done
 done
done


