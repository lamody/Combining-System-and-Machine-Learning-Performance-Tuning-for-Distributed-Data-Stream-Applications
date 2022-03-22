
Dependencies
------------
Java 1.8
Scala 2.11
sbt 0.13.18
Spark 2.3.2


Building
--------
sbt clean
sbt compile
sbt package     # Will create a skinny jar file without dependencies
sbt assembly    # Will create a fat jar file with dependencies


Executing with arff files
-------------------------
# Note 1: assumes $SPARK_HOME is set
# Note 2: option -f in SGDLearner must match the number of features

./spark.sh "EvaluatePrequential -l (org.apache.spark.streamdm.classifiers.SGDLearner -f 8 -o LogisticLoss) -s (FileReader -k 100 -i 45000 -f /home/hero/deploy/streamDM/data/electNormNew.arff) -e (aggression.evaluation.ExtendedClassificationEvaluator) -h" 1>results.txt 2>debug.log

./spark.sh "EvaluatePrequential -l (org.apache.spark.streamdm.classifiers.trees.HoeffdingTree -c 0.1) -s (FileReader -k 100 -i 1000 -d 1000 -f /home/hero/deploy/streamDM/data/electNormNew.arff) -e (aggression.evaluation.ExtendedClassificationEvaluator) -h" 1>results.txt 2>debug.log

./spark.sh "1000 EvaluatePrequential -l (org.apache.spark.streamdm.classifiers.trees.HoeffdingTree -c 0.1) -s (FileReader -k 400 -i 10000 -d 200 -f ../../data/100k_tweets/arff/labeled_tweets_prepro_norm.arff) -e (aggression.evaluation.ExtendedClassificationEvaluator) -h" 1>results_ht_pre_norm.txt 2>debug_ht_pre_norm.log


Executing with tweet files
-------------------------
# Note 1: assumes $SPARK_HOME is set
# Note 2: option -f in SGDLearner must match the number of features

./spark.sh "aggression.tasks.EvaluateTweetAggression -t (aggression.tweet.TweetFileReader -k 200 -i 600 -d 1000 -f ../../data/labeled_tweets.txt) -f (aggression.tweet.FeatureExtractor -f ../../data/100k_tweets/arff/labeled_tweets_preprocess.arff.head -p -a) -l (org.apache.spark.streamdm.classifiers.trees.HoeffdingTree -c 0.01) -h" 1>results_ht.txt 2>debug_ht.log

./spark.sh "aggression.tasks.EvaluateTweetAggression -t (aggression.tweet.TweetFileReader -k 100 -i 200 -d 1000 -f ../../data/labeled_tweets.txt) -f (aggression.tweet.FeatureExtractor -f ../../data/100k_tweets/arff/labeled_tweets_preprocess.arff.head -p -a) -l (org.apache.spark.streamdm.classifiers.MultiClassLearner -l (org.apache.spark.streamdm.classifiers.SGDLearner -f 17 -o LogisticLoss)) -h" 1>results_mc.txt 2>debug_mc.log

# Note: in spark.sh, the number of threads n (in local[n])
#       must be greater than the number of files in the dir
~/deploy/hadoop-3.1.2/bin/hdfs dfs -mkdir /user/hero/tweets
~/deploy/hadoop-3.1.2/bin/hdfs dfs -put ../../data/labeled_tweets.txt /user/hero/tweets
./spark.sh "aggression.tasks.EvaluateTweetAggression -t (aggression.tweet.TweetDirReader -p hdfs://localhost:9000/user/hadoop/input -c 1000 -r 6) -f (aggression.tweet.FeatureExtractor -f ../../data/100k_tweets/arff/tweets_p1_n0_b0_a1_nosp_f16.arff.head -p -n 2) -l (org.apache.spark.streamdm.classifiers.trees.HoeffdingTree  -c 0.01) -h" 1>results_ht.txt 2>debug_ht.log


Most important command line arguments
-------------------------------------
Usage: ./spark.sh "[batch_interval] main_class [class_options]"
Available main classes:
   aggression.tasks.EvaluateTweetAggression
   org.apache.spark.streamdm.tasks.EvaluatePrequential
   org.apache.spark.streamdm.tasks.EvaluateOutlierDetection
   org.apache.spark.streamdm.tasks.ClusteringTrainEvaluate

Available options for aggression.tasks.EvaluateTweetAggression:
   -t tweetReader (default: aggression.tweet.TweetFileReader)
   -f featureExtractor (default: aggression.tweet.FeatureExtractor)
   -l learner (default: org.apache.spark.streamdm.classifiers.SGDLearner) - others: MultiClassLearner, HoeffdingTree
   -e evaluator (default: BasicClassificationEvaluator)
   -w resultsWriter (default: PrintStreamWriter)
   -h shouldPrintHeader - Whether or not to print the evaluator header on the output file

Available options for aggression.tweet.TweetFileReader:
   -k chunkSize (default: 10000)
   -i instanceLimit (default: 100000)
   -d slideDuration (default: 100) - in milliseconds
   -f fileName

Available options for aggression.tweet.TweetDirReader:
   -p path
   -c maxCount - max tweets per partition (default: 1000)
   -r maxReceivers - max number of receivers (default: 0 - unlimited)
   -l rateLimit - max partitions per sec (default: 0 - unlimited)

Available options for aggression.tweet.FeatureExtractor:
   -f fileName with feature definitions and stats in .arff format
   -p enablePreprocessing flag
   -a enableAdaptiveBoW flag
   -n normalization (default: 1) - 0=none, 1=normalize, 2=normalize-no-outlier, 3=standardize

Available options for org.apache.spark.streamdm.classifiers.trees.HoeffdingTree:
   -n numericObserverType (default: 0) - 0: gaussian // not used
   -s splitCriterion (default: InfoGainSplitCriterion) - others: GiniSplitCriterion, VarianceReductionSplitCriterion
   -o growthAllowed - Allow to grow
   -b binaryOnly - Only allow binary splits
   -g numGrace (default: 200) - number of examples a leaf should observe between split attempts
   -t tieThreshold (default: 0.05) - Threshold below which a split will be forced to break ties.
   -c splitConfidence (default: 1.0E-7) - allowable error in split decision
   -l learningNodeType (default: 2) - 0=ActiveLearningNode, 1=LearningNodeNB, 2=LearningNodeNBAdaptive
   -q nbThreshold (default: 0) - number of examples a leaf should observe between permitting Naive Bayes
   -p noPrePrune - Disable pre-pruning
   -r removePoorFeatures
   -h MaxDepth (default: 20)

Available options for org.apache.spark.streamdm.classifiers.MultiClassLearner:
   -l baseClassifier (default: SGDLearner)

Available options for org.apache.spark.streamdm.classifiers.SGDLearner:
   -f numFeatures (default: 3)
   -l lambda (default: 0.01)
   -o lossFunction (default: LogisticLoss) - others: PerceptronLoss, HingeLoss, SquaredLoss
   -r regularizer (default: ZeroRegularizer) - others: L1Regularizer, L2Regularizer
   -p regParam (default: 0.001) - Regularization parameter


