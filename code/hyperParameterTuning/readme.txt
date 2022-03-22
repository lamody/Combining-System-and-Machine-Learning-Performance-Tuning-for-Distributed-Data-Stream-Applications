Arff Files
==========
HT
==
./run-tuning-spark-ht.sh ../../data/100k_tweets/arff/tweets_p1_n2_b0_a0.arff ../../data/HPO/arff-file/HT/<conf>/
./run-tuning-spark-ht.sh ../../data/100k_tweets/arff/tweets_p1_n2_b0_a0.arff ../../data/HPO/arff-file/HT/1/

SGD
===
./run-tuning-spark-sgd.sh ../../data/100k_tweets/arff/tweets_p1_n2_b0_a0.arff ../../data/HPO/arff-file/SGD/<conf>/
./run-tuning-spark-sgd.sh ../../data/100k_tweets/arff/tweets_p1_n2_b0_a0.arff ../../data/HPO/arff-file/SGD/1/

Agrawal
=======
HT
==
./run-tuning-spark-ht.sh ../../data/agrawal/agrawal_f6_p02.arff ../../data/HPO/agrawal/HT/1/
./run-tuning-spark-sgd.sh ../../data/agrawal/agrawal_f6_p02.arff ../../data/HPO/agrawal/SGD/1/

Tweet Files
===========
HT
==
./run-tuning-spark-ht-tweets-file.sh ../../data/labeled_tweets_bin_nosp.txt ../../data/100k_tweets/arff/tweets_p1_n2_b1_a0_nosp.arff.head ../../data/HPO/tweets-file/HT/<conf>/
./run-tuning-spark-ht-tweets-file.sh ../../data/labeled_tweets_bin_nosp.txt ../../data/100k_tweets/arff/tweets_p1_n2_b1_a0_nosp.arff.head ../../data/HPO/tweets-file/HT/1/

SGD
===
./run-tuning-spark-sgd-tweets-file.sh ../../data/labeled_tweets_bin_nosp.txt ../../data/100k_tweets/arff/tweets_p1_n2_b1_a0_nosp.arff.head ../../data/HPO/tweets-file/SGD/<conf>/
./run-tuning-spark-sgd-tweets-file.sh ../../data/labeled_tweets_bin_nosp.txt ../../data/100k_tweets/arff/tweets_p1_n2_b1_a0_nosp.arff.head ../../data/HPO/tweets-file/SGD/1/

Cluster
=======
HT
==
./process-tweets-spark-ht.sh hdfs://dicl15.cut.ac.cy:9000/user/hadoop/inputbin4 ../../data/100k_tweets/arff/tweets_p1_n2_b1_a0_nosp.arff.head ../../data/HPO/HT/

SGD
===
./process-tweets-spark-sgd.sh hdfs://dicl15.cut.ac.cy:9000/user/hadoop/inputbin4 ../../data/100k_tweets/arff/tweets_p1_n2_b1_a0_nosp.arff.head ../../data/HPO/SGD/


Agrawal
=======
HT
==
./process-agrawal-spark-ht.sh ../../data/agrawal/cluster-dir-test/ ../../data/agrawal/agrawal_f6_p02.arff.head ../../data/HPO/agrawal/cluster/HT/

SGD
===
./process-agrawal-spark-sgd.sh ../../data/agrawal/cluster-dir-test/ ../../data/agrawal/agrawal_f6_p02.arff.head ../../data/HPO/agrawal/cluster/SGD/

RF
==
./process-agrawal-spark-rf.sh hdfs://dicl15.cut.ac.cy:9000/user/hadoop/agrawal4/ ../../data/agrawal/agrawal_f6_p0/agrawal_f6_p0.arff.head ../../data/HPO/agrawal/agrawal_f6_p0/RF/