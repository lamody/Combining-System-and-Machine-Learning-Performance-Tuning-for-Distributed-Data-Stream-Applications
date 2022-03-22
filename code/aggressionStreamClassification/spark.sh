#!/usr/bin/env bash

script=$(readlink -f "$0")
basedir=$(dirname "$script")
# master="spark://localhost:7077"
master="spark://dicl17.cut.ac.cy:7077"
mode="standalone"

if [ $mode = "local" ]; then
  $SPARK_HOME/bin/spark-submit \
    --class "aggression.aggressionStreamJob" \
    --master local[4] \
    ${basedir}/target/scala-2.11/aggression-stream-classification-assembly-0.1.jar \
    $@
elif [ $mode = "standalone" ]; then
  $SPARK_HOME/bin/spark-submit \
    --class "aggression.aggressionStreamJob" \
    --master ${master} \
    ${basedir}/target/scala-2.11/aggression-stream-classification-assembly-0.1.jar \
    $@
else
  echo "Invalid mode ${mode}. Must be either local or standalone."
fi
