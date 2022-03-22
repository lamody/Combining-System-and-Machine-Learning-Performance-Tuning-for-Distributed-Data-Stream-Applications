HT
==
python3 ht.py /home/lamody/streamingaggression/data/HPO/HT/1/tweets_p1_n2_b0_a0/ /home/lamody/streamingaggression/data/HPO/HT/2/tweets_p1_n2_b0_a0/ /home/lamody/streamingaggression/data/HPO/HT/3/tweets_p1_n2_b0_a0/ /home/lamody/streamingaggression/data/HPO/HT/

SGD
===
python3 sgd.py /home/lamody/streamingaggression/data/HPO/SGD/1/tweets_p1_n2_b0_a0/ /home/lamody/streamingaggression/data/HPO/SGD/2/tweets_p1_n2_b0_a0/ /home/lamody/streamingaggression/data/HPO/SGD/3/tweets_p1_n2_b0_a0/ /home/lamody/streamingaggression/data/HPO/SGD/
python3 sgd.py /home/lamody/streamingaggression/data/HPO/SGD/5k_chunkSize/1/tweets_p1_n2_b0_a0/ /home/lamody/streamingaggression/data/HPO/SGD/5k_chunkSize/2/tweets_p1_n2_b0_a0/ /home/lamody/streamingaggression/data/HPO/SGD/5k_chunkSize/3/tweets_p1_n2_b0_a0/ /home/lamody/streamingaggression/data/HPO/SGD/5k_chunkSize/

Compare
=======
python3 compare.py <first result> <second result> <first result label> <second result label> <conf column> <accuracy column> <overall accuracy column> <job time column> <task time column> <output dir>
python3 compare.py /home/lamody/streamingaggression/data/HPO/SGD/2k_chunkSize/ /home/lamody/streamingaggression/data/HPO/SGD/5k_chunkSize/ "2k Chunk Size" "5k Chunk Size" 4 5 7 9 11 /home/lamody/streamingaggression/data/HPO/SGD/compare/

Some samples
------------
Comparing:
 - 2k-chunkSize_local-standalone  
 - 5k-chunkSize_local-standalone  
 - local_2k-5k-chunkSize  
 - standalone_2k-5k-chunkSize 

python3 compare.py ../../data/HPO/tweets/SGD/local/2k_chunkSize/results/ ../../data/HPO/tweets/SGD/standalone/2k_chunkSize/results/ "Local mode" "Standalone mode" 4 5 7 9 11 ../../data/HPO/tweets/SGD/compare/2k-chunkSize_local-standalone/
python3 compare.py ../../data/HPO/tweets/SGD/local/5k_chunkSize/results/ ../../data/HPO/tweets/SGD/standalone/5k_chunkSize/results/ "Local mode" "Standalone mode" 4 5 7 9 11 ../../data/HPO/tweets/SGD/compare/5k-chunkSize_local-standalone/
python3 compare.py ../../data/HPO/tweets/SGD/local/2k_chunkSize/results/ ../../data/HPO/tweets/SGD/local/5k_chunkSize/results/ "2k Chunk Size" "5k Chunk Size" 4 5 7 9 11 ../../data/HPO/tweets/SGD/compare/local_2k-5k-chunkSize/
python3 compare.py ../../data/HPO/tweets/SGD/standalone/2k_chunkSize/results/ ../../data/HPO/tweets/SGD/standalone/5k_chunkSize/results/ "2k Chunk Size" "5k Chunk Size" 4 5 7 9 11 ../../data/HPO/tweets/SGD/compare/standalone_2k-5k-chunkSize/

Cluster
=======
python3 generate-cluster-graphs.py HT ~/streamingaggression/data/HPO/tweets/HT/ ~/streamingaggression/data/HPO/paper-results/tweets/HT/
python3 generate-cluster-graphs.py SGD ~/streamingaggression/data/HPO/tweets/SGD/ ~/streamingaggression/data/HPO/paper-results/tweets/SGD/
python3 generate-cluster-graphs.py HT ~/streamingaggression/data/HPO/agrawal/agrawal_f6_p0/HT/ ~/streamingaggression/data/HPO/paper-results/agrawal/HT/
python3 generate-cluster-graphs.py SGD ~/streamingaggression/data/HPO/agrawal/agrawal_f6_p0/SGD/ ~/streamingaggression/data/HPO/paper-results/agrawal/SGD/

SCP
===
scp -r -P 3440 hadoop@diclgw.cut.ac.cy:/home1/hadoop/streamingaggression/data/HPO/paper-results/ /home/lamody/Desktop/