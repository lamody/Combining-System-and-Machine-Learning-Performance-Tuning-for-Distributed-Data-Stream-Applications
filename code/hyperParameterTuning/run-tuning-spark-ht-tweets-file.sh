#!/usr/bin/env bash

# set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Perform hyper-parameter tuning for Spark's hoeffding tree

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

declare -a splitconfs=("0.01" "0.1") # "0.5")
declare -a tiethres=("0.05" "0.1")
declare -a graces=("200" "500")
declare -a splitcrits=("InfoGainSplitCriterion" "GiniSplitCriterion")
declare -a depths=("20" "30")

#################################################
# Get input parameters
data_file=$1
header_file=$2
output_dir=$3

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

preprocess="-p"
normalization="2"
adaptive=""

iter=1
total=$((${#splitconfs[@]} * ${#tiethres[@]} * ${#graces[@]} * ${#splitcrits[@]} * ${#depths[@]}))

cp /home1/hadoop/deploy/spark/conf/spark-defaults.conf ${output_dir}/spark-defaults.conf
cp /home1/hadoop/deploy/spark/conf/spark-env.sh ${output_dir}/spark-env.sh

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
                    out_filename="spark_ht_c${splitconf}_t${tiethre}_g${grace}_s${splitcrit}_d${depth}"
                    echo "${out_filename} ($iter/$total)"

                    ${run_spark} "aggression.tasks.EvaluateTweetAggression" \
                        " -t (aggression.tweet.TweetFileReader -k ${chunkSize} -i ${instanceLimit} -d ${slideDuration} -f ${data_file})" \
                        " -f (aggression.tweet.FeatureExtractor -f ${header_file} ${preprocess} -n ${normalization} ${adaptive})" \
                        " -l (org.apache.spark.streamdm.classifiers.trees.HoeffdingTree -c ${splitconf} -t ${tiethre} -g ${grace} -s ${splitcrit} -h ${depth})" \
                        " -e (aggression.evaluation.ExtendedClassificationEvaluator) -h " \
                        1> "${output_dir}/${out_filename}.csv" 2> "${output_dir}/${out_filename}.log"

                    iter=$((iter+1))

                    # latest=$(ls /opt/spark/logs/ -t | head -n 1)
                    # ln -s "/opt/spark/logs/${latest}" "${output_dir}/${out_filename}"

                    latest=$(/home1/hadoop/deploy/hadoop/bin/hdfs dfs -ls -t -C spark-events | head -n 1)
                    /home1/hadoop/deploy/hadoop/bin/hdfs dfs -copyToLocal ${latest} "${output_dir}/${out_filename}"
                done
            done
        done
    done
done


