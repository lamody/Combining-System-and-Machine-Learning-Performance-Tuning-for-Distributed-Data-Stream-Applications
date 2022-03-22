#!/usr/bin/env bash

set -e            # Stop the script if some command fails
set -o nounset    # Check for unbound variables
set -o pipefail   # Fail if some part of a pipe fails

# Remove one group of features from the input data at a time
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

# Create the feature groups
declare -a group_names
declare -a group_counts
declare -a group_features

#group_names+=( "allbutbow" )
#group_counts+=( 15 )
#group_features+=( "cntLists" "cntPosts" "accountAge" "numHashtags" "numUrls" "numUpperCases" "wordsPerSentence" "meanWordLength" "cntAdjective" "cntAdverbs" "cntVerbs" "sentimentScorePos" "sentimentScoreNeg" "cntFollowers" "cntFriends" )

group_names+=( "user" )
group_counts+=( 3 )
group_features+=( "cntLists" "cntPosts" "accountAge" )

group_names+=( "tweet" )
group_counts+=( 2 )
group_features+=( "numHashtags" "numUrls" )

group_names+=( "text" )
group_counts+=( 3 )
group_features+=( "numUpperCases" "wordsPerSentence" "meanWordLength" )

group_names+=( "pos" )
group_counts+=( 3 )
group_features+=( "cntAdjective" "cntAdverbs" "cntVerbs" )

group_names+=( "sentiment" )
group_counts+=( 2 )
group_features+=( "sentimentScorePos" "sentimentScoreNeg" )

group_names+=( "network" )
group_counts+=( 2 )
group_features+=( "cntFollowers" "cntFriends" )

group_names+=( "bow" )
group_counts+=( 1 )
group_features+=( "cntSwearWords" )

# Process input parameters
data_file=$1
output_dir=$2

# Create output directory
filename="$(basename ${data_file})"
filename="${filename%.*}"
output_dir=${output_dir%/}
output_dir="${output_dir}/${filename}"

mkdir -p ${output_dir}

# Remove the features, one group at a time, and run moa
index=0
cnt=${#group_names[@]}
for (( i = 0 ; i < cnt ; i++ ))
do
  features=( ${group_features[@]:${index}:${group_counts[$i]}} )
  index=$(($index + ${group_counts[$i]}))
  echo "Group [$i]: ${group_names[$i]}"
  echo "Features : ${features[@]}"

  out_filename="moa_ht_c${splitconf}_t${tiethre}_g${grace}_s${splitcrit}_b${binsplit}_p${nopreprune}_no${group_names[$i]}"
  echo ${out_filename}

  python ${REMOVE_FEATURES_PY} "${data_file}" "${data_file}_no${group_names[$i]}.arff" "${features[@]}"

  java -Xmx${MEMORY} \
       -cp "${MOA_DIR}/lib/moa.jar:${MOA_DIR}/lib/*" \
       -javaagent:${MOA_DIR}/lib/sizeofag-1.0.4.jar moa.DoTask \
       "EvaluatePrequential -l (trees.HoeffdingTree -c ${splitconf} -t ${tiethre} -g ${grace} -s ${splitcrit} ${binsplit} ${nopreprune})" \
       " -s (ArffFileStream -f ${data_file}_no${group_names[$i]}.arff)" \
       " -e (BasicClassificationPerformanceEvaluator -o -p -r -f)" \
       " -i 86000 -f ${frequency} -w ${width}" 1> "${output_dir}/${out_filename}.csv" 2> "${output_dir}/${out_filename}.log"

  rm "${data_file}_no${group_names[$i]}.arff"
  echo ""

done


