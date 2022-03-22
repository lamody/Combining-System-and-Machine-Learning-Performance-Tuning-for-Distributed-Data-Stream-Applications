from math import ceil, floor
import os
import sys
import csv
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def fileIterator(model, dataset):
    if model == 'HT':
        prefix = 'spark_ht'
        if dataset == "agrawal":
            for c in ['0.001', '0.01', '0.1']:
                for t in ['0.001', '0.05', '0.1']:
                    for g in ['50', '800', '1500']:
                        for s in ['InfoGainSplitCriterion', 'GiniSplitCriterion']:
                            for d in ['20', '30']:
                                yield "{}_c{}_t{}_g{}_s{}_d{}".format(prefix, c, t, g, s, d)
        elif dataset == "tweets":
            for c in ['0.01', '0.1']:
                for t in ['0.05', '0.1']:
                    for g in ['200', '500']:
                        for s in ['InfoGainSplitCriterion', 'GiniSplitCriterion']:
                            for d in ['20', '30']:
                                yield "{}_c{}_t{}_g{}_s{}_d{}".format(prefix, c, t, g, s, d)
    elif model == 'SGD':
        prefix = 'spark_sgd'
        for l in ["0.001", "0.01", "0.05", "0.1"]:
            for o in ["LogisticLoss", "SquaredLoss", "HingeLoss", "PerceptronLoss"]:
                for r in ["ZeroRegularizer", "L2Regularizer"]:
                    for p in ["0.001", "0.01"]:
                        yield "{}_l{}_o{}_r{}_p{}".format(prefix, l, o, r, p)
    elif model == 'RF':
        prefix = 'spark_rf'
        for e in ['10']:
            for m in ['5']:
                for a in ['6']:
                    for c in ['0.01']:
                        for t in ['0.1']:
                            for g in ['1500']:
                                for s in ['InfoGainSplitCriterion']:
                                    for d in ['20']:
                                        yield "{}_es{}_m{}_a{}_c{}_t{}_g{}_s{}_d{}".format(prefix, e, m, a, c, t, g, s, d)

def getData(fileName):
    instances = []
    accuracy = []
    recall = []
    precission = []
    f1Score = []

    fp = open(fileName, 'r')
    next(fp)
    
    for line in fp:
        instances.append(int(float(line.split(',')[1])))
        accuracy.append(float(line.split(',')[2]) * 100)
        recall.append(float(line.split(',')[3]) * 100)
        precission.append(float(line.split(',')[4]) * 100)
        f1Score.append(float(line.split(',')[5]) * 100)

    fp.close()
    return np.sum(instances), np.average(accuracy), np.average(recall), np.average(precission), np.average(f1Score)

def runSparkLogParser(fileName, outputFile):
    # Local
    # subprocess.run(['python2', '/home/lamody/Spark-Log-Parser/main.py', fileName, outputFile])
    # Cluster
    subprocess.run(['python2', '/home1/hadoop/Spark-Log-Parser-master/main.py', fileName, outputFile])

def parseSparkLogParser(inputFile):
    jobs = 0
    jobTime = 0
    tasks = 0
    taskTime = 0 

    job = False
    task = False

    stages = []
    stageTasks = []
    jobID = None
    isReceiver = False
    stagesToIgnore = []

    taskID = None

    with open(inputFile, 'r') as fp:
        for line in fp:
            if not job and '---> Jobs <---' in line:
                job = True
            
            if job:
                if jobID != None and line.strip() == "":
                    if any([x == 0 for x in stageTasks]) or isReceiver:
                        stagesToIgnore = stagesToIgnore + stages
                    else:
                        jobs = jobs + 1
                        jobTime = jobTime + time

                    jobID = None
                    stages = []
                    stageTasks = []                
                    isReceiver = False

                    continue

                elif 'Run time:' in line:
                    time = line.strip().split(': ')[1]
                    time = int(time.strip()[:-2])
                
                elif 'Stage ' in line:
                    stage = int(line.strip().split(" ")[-1].split("(")[1].split(")")[0])
                    stages.append(stage)

                elif "Number of tasks" in line:
                    tasks = int(line.strip().split(": ")[1])
                    stageTasks.append(tasks)

                elif "Job " in line:
                    jobID = int(line.strip().split(" ")[1])

                elif "RDD Receiver" in line:
                    isReceiver = True

            if not task and '---> Tasks <---' in line:
                job = False
                task = True

            if task:
                if taskID != None and line.strip() == "":
                    if stage not in stagesToIgnore:
                        tasks = tasks + 1
                        taskTime = taskTime + time

                    taskID = None
                    continue

                elif "stage:" in line:
                    taskID = int(line.strip().split(" ")[1])
                    stage = int(line.strip().split("stage: ")[1].split(",")[0])

                elif "Run time:" in line:
                    time = line.strip().split(': ')[1]
                    time = int(time.strip()[:-2])

    return jobs, jobTime, tasks, taskTime

def normalize(value, min, max):
    return (value - min) / (max - min) * 100

def inverseNormalize(value, min, max):
    return (max - value) / (max - min) * 100

def hpsKeys(hps):
    r = hps.split("_")[2:]
    return [x[0] for x in r]

def hpsValues(hps):
    r = hps.split("_")[2:]
    return [x[1:] for x in r]

def writeStatsToCSVFile(outputDir, hps, stats, normalizedStats, times, normalizedTimes):
    x, y, _ = np.shape(stats)
    with open(os.path.join(outputDir, "results.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(
            ["ExpID", *hpsKeys(hps[0]), "Instances", 
            "Accuracy (%)", "Normalized Accuracy (%)",  
            "Recall (%)", "Normalized Recall (%)", 
            "Precision (%)", "Normalized Precision (%)", 
            "F1-Score (%)", "Normalized F1-Score (%)", 
            "Jobs (Count)", "Job Time (ms)", "Normalized Job Time (%)", 
            "Tasks (Count)", "Task Time (ms)", "Normalized Task Time (%)", 
            "Normalized Time Joined (%)"]
        )
        for i in range(x):
            expID = "s{}".format(i + 1)
            for j in range(y):
                hp = hpsValues(hps[j])
                instances = stats[i,j,0]
                accuracy = stats[i,j,1]
                normalizedAccuracy = normalizedStats[i,j,1]
                recall = stats[i,j,2]
                normalizedRecall = normalizedStats[i,j,2]
                precision =  stats[i,j,3]
                normalizedPrecision = normalizedStats[i,j,3]
                f1Score =  stats[i,j,4]
                normalizedF1Score = normalizedStats[i,j,4]
                jobs = times[i,j,0]
                jobTime = times[i,j,1]
                normalizedJobTime = normalizedTimes[i,j,1]
                tasks = times[i,j,2]
                taskTime = times[i,j,3]
                normalizedTaskTime = normalizedTimes[i,j,3]
                averageNormalizedTimeJoined = (normalizedJobTime + normalizedTaskTime) / 2

                writer.writerow(
                    [expID] + hp + [instances] + 
                    [accuracy, normalizedAccuracy] + 
                    [recall, normalizedRecall] +
                    [precision, normalizedPrecision] +
                    [f1Score, normalizedF1Score] +
                    [jobs, jobTime, normalizedJobTime] +
                    [tasks, taskTime, normalizedTaskTime] + 
                    [averageNormalizedTimeJoined]
                )

def generateGraphs(outputDir, hps, stats, normalizedStats, times, normalizedTimes, throughputs, normalizedThroughputs, latencies, normalizedLatencies):
    x, y, _ = np.shape(stats)
    
    setupGroups = [
        ['s1', 's2', 's3', 's4'],
        ['s5', 's2', 's6', 's7'],
        ['s8', 's6', 's3', 's9']
    ]

    # list of tuples (records/partition, partitions/second)
    setupLabels = [
        (100, 16),
        (200, 16),
        (300, 16),
        (400, 16),
        (200, 8),
        (200, 24),
        (200, 32),
        (100, 48),
        (400, 12),
    ]

    jobTimeStep = ceil(np.max(times[:,:,1]) / 10)
    jobTimeTicks = np.around((np.arange(0, np.max(times[:,:,1]) + 1000, jobTimeStep) / 1000), 1)
    taskTimeStep = ceil(np.max(times[:,:,3]) / 10)
    taskTimeTicks = np.around((np.arange(0, np.max(times[:,:,3]) + 1000, taskTimeStep) / 1000), 1)

    throughputStep = ceil(np.max(throughputs) / 10)
    throughputTicks = np.around((np.arange(0, np.max(throughputs) + 10, throughputStep) / 1000), 1)
    latencyStep = ceil(np.max(latencies) / 10)
    latencyTicks = np.around((np.arange(0, np.max(latencies) + 10, latencyStep)), 1)
    hyperParameterStep = ceil(y / 20)

    # Line plots
    for i in range(x):
        plt.plot(np.arange(1, y + 1), stats[i,:,2], label="r={}, p={}".format(*setupLabels[i]))
    
    plt.legend()
    plt.xlim(1, y)
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Recall Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Recall (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Recall Plot')))
    plt.clf()

    for i in range(x):
        plt.plot(np.arange(1, y + 1), stats[i,:,3], label="r={}, p={}".format(*setupLabels[i]))
    
    plt.legend()
    plt.xlim(1, y)
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Precision Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Precision (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Precision Plot')))
    plt.clf()

    for i in range(x):
        plt.plot(np.arange(1, y + 1), stats[i,:,4], label="r={}, p={}".format(*setupLabels[i]))
    
    plt.legend()
    plt.xlim(1, y)
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('F1-Score Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('F1-Score (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'F1-Score Plot')))
    plt.clf()

    for i in range(x):
        plt.plot(np.arange(1, y + 1), normalizedStats[i,:,1], label="r={}, p={}".format(*setupLabels[i]))
    
    plt.legend()
    plt.xlim(1, y)
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Normalized Accuracy Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Normalized Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Normalized Accuracy Plot')))
    plt.clf()

    for i in range(x):
        plt.plot(np.arange(1, y + 1), normalizedStats[i,:,2], label="r={}, p={}".format(*setupLabels[i]))
    
    plt.legend()
    plt.xlim(1, y)
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Normalized Recall Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Normalized Recall (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Normalized Recall Plot')))
    plt.clf()

    for i in range(x):
        plt.plot(np.arange(1, y + 1), normalizedStats[i,:,3], label="r={}, p={}".format(*setupLabels[i]))
    
    plt.legend()
    plt.xlim(1, y)
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Normalized Precision Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Normalized Precision (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Normalized Precision Plot')))
    plt.clf()

    for i in range(x):
        plt.plot(np.arange(1, y + 1), normalizedStats[i,:,4], label="r={}, p={}".format(*setupLabels[i]))
    
    plt.legend()
    plt.xlim(1, y)
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Normalized F1-Score Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Normalized F1-Score (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Normalized F1-Score Plot')))
    plt.clf()

    # Scatter Plots
    for i in range(x):
        plt.scatter([i + 1] * y, normalizedStats[i,:,1], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Configuration Normalized Accuracy Scatter Plot')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Normalized Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Normalized Accuracy Scatter Plot')))
    plt.clf()

    for i in range(x):
        plt.scatter(normalizedStats[i,:,1], (normalizedTimes[i,:,1] + normalizedTimes[i,:,3]) / 2, label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(0, 101, 10))
    plt.xlim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Normalized Accuracy Normalized Joined Times Scatter Plot')
    plt.xlabel('Normalized Accuracy (%)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Normalized Accuracy Normalized Joined Times Scatter Plot')))
    plt.clf()

    for i in range(x):
        plt.scatter([i + 1] * y, (normalizedTimes[i,:,1] + normalizedTimes[i,:,3]) / 2, label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Configuration Normalized Joined Times Scatter Plot')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Normalized Joined Times Scatter Plot')))
    plt.clf()

    for i in range(x):
        plt.scatter(np.arange(1, y + 1), normalizedStats[i,:,1], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.xlim(0, y + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Experiment Normalized Accuracy Scatter Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Normalized Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Normalized Accuracy Scatter Plot')))
    plt.clf()

    for i in range(x):
        plt.scatter(np.arange(1, y + 1), (normalizedTimes[i,:,1] + normalizedTimes[i,:,3]) / 2, label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.xlim(0, y + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Experiment Normalized Joined Times Scatter Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Normalized Joined Times Scatter Plot')))
    plt.clf()

    for i in range(x):
        plt.scatter([i + 1] * y, times[i,:,1], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, np.max(times[:,:,1]) + 10, jobTimeStep), jobTimeTicks)
    plt.ylim(0, np.max(times[:,:,1]) + 10)
    # plt.title('Configuration Job Times Scatter Plot')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Job Times (sec)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Job Times Scatter Plot')))
    plt.clf()

    for i in range(x):
        plt.scatter([i + 1] * y, normalizedTimes[i,:,1], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Configuration Normalized Job Times Scatter Plot')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Normalized Job Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Normalized Job Times Scatter Plot')))
    plt.clf()

    for i in range(x):
        plt.scatter(np.arange(1, y + 1), times[i,:,0], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.xlim(0, y + 1)
    plt.yticks(np.arange(0, np.max(times[:,:,0]) + 10, 50))
    plt.ylim(0, np.max(times[:,:,0]) + 10)
    # plt.title('Experiment Job Count Scatter Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Job Count (#)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Job Count Scatter Plot')))
    plt.clf()

    for i in range(x):
        plt.scatter([i + 1] * y, times[i,:,3], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, np.max(times[:,:,3]) + 10, taskTimeStep), taskTimeTicks)
    plt.ylim(0, np.max(times[:,:,3]) + 10)
    # plt.title('Configuration Task Times Scatter Plot')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Task Times (sec)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Task Times Scatter Plot')))
    plt.clf()

    for i in range(x):
        plt.scatter([i + 1] * y, normalizedTimes[i,:,3], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Configuration Normalized Task Times Scatter Plot')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Normalized Task Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Normalized Task Times Scatter Plot')))
    plt.clf()

    for i in range(x):
        plt.scatter(np.arange(1, y + 1), times[i,:,2], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.xlim(0, y + 1)
    plt.yticks(np.arange(0, np.max(times[:,:,2]) + 10, 1000))
    plt.ylim(0, np.max(times[:,:,2]) + 10)
    # plt.title('Experiment Task Count Scatter Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Task Count (#)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Task Count Scatter Plot')))
    plt.clf()

    # Box Plots
    plt.boxplot([stats[i,:,1] for i in range(x)])
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Whisker Plot (Accuracy)')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (Accuracy)')))
    plt.clf()

    plt.boxplot([normalizedStats[i,:,1] for i in range(x)])
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Whisker Plot (Normalized Accuracy)')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Normalized Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (Normalized Accuracy)')))
    plt.clf()

    plt.boxplot([(normalizedTimes[i,:,1] + normalizedTimes[i,:,3]) / 2 for i in range(x)])
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Whisker Plot (Normalized Joined Times)')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (Normalized Joined Times)')))
    plt.clf()

    plt.boxplot([0.5 * normalizedStats[i,:,1] + 0.5 * (normalizedTimes[i,:,1] + normalizedTimes[i,:,3]) / 2 for i in range(x)])
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Whisker Plot (50% Normalized Accuracy, 50% Normalized Joined Times)')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Score (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (50% Normalized Accuracy, 50% Normalized Joined Times)')))
    plt.clf()

    plt.boxplot([normalizedStats[i,:,1] / 3 + normalizedTimes[i,:,1] / 3 + normalizedTimes[i,:,3] / 3 for i in range(x)])
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Whisker Plot (Equally Wighted - Normalized Accuracy, Job and Task Times)')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Score (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (Equally Wighted - Normalized Accuracy, Normalized Job and Normalized Task Times)')))
    plt.clf()
    
    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System Configuration", labelpad=25)
    plt.ylabel("Accuracy (%)")
    plt.subplots_adjust(bottom=0.25)
        
    bestHP = np.argmax(stats[0,:,1])

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            axs[i].scatter([j + 1] * y, stats[int(g[1]) - 1,:,1], label=g, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])
            axs[i].scatter([j + 1], stats[int(g[1]) - 1,bestHP,1], color="black")

        plt.sca(axs[i])
        plt.xticks([])
        plt.xlim(0.5, 4.5)
        plt.yticks(np.arange(0, 101, 10))
        plt.ylim(0, 100)

        indices = [int(c[1]) - 1 for c in s]
        cells = [[setupLabels[i][0] for i in indices], [setupLabels[i][1] for i in indices]]
        labels = ["  r  ", "  p  "] if i == 0 else None

        plt.table(
            bbox=[0, -0.17, 1, 0.12],
            rowLabels=labels,
            cellText=cells
        )
    
    plt.savefig(os.path.join(os.path.join(outputDir, 'Grouped Cluster Configurations Accuracy')))
    plt.clf()


    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System Configuration", labelpad=25)
    plt.ylabel("Job Times (sec)", labelpad=10)
    plt.subplots_adjust(bottom=0.25)
        
    bestHP = np.argmin(times[0,:,1])

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            axs[i].scatter([j + 1] * y, times[int(g[1]) - 1,:,1], label=g, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])
            axs[i].scatter([j + 1], times[int(g[1]) - 1,bestHP,1], color="black")

        plt.sca(axs[i])
        plt.xticks([])
        plt.xlim(0.5, 4.5)
        plt.yticks(np.arange(0, np.max(times[:,:,1]) + 10, jobTimeStep), jobTimeTicks)
        plt.ylim(0, np.max(times[:,:,1]) + 10)

        indices = [int(c[1]) - 1 for c in s]
        cells = [[setupLabels[i][0] for i in indices], [setupLabels[i][1] for i in indices]]
        labels = ["  r  ", "  p  "] if i == 0 else None

        plt.table(
            bbox=[0, -0.17, 1, 0.12],
            rowLabels=labels,
            cellText=cells
        )
    
    plt.savefig(os.path.join(os.path.join(outputDir, 'Grouped Cluster Configurations Job Times')))
    plt.clf()

    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System Configuration", labelpad=25)
    plt.ylabel("Task Times (sec)", labelpad=20)
    plt.subplots_adjust(left=0.15, bottom=0.25)

    bestHP = np.argmin(times[0,:,3])

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            axs[i].scatter([j + 1] * y, times[int(g[1]) - 1,:,3], label=g, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])
            axs[i].scatter([j + 1], times[int(g[1]) - 1,bestHP,3], color="black")

        plt.sca(axs[i])
        plt.xticks([])
        plt.xlim(0.5, 4.5)
        plt.yticks(np.arange(0, np.max(times[:,:,3]) + 10, taskTimeStep), taskTimeTicks)
        plt.ylim(0, np.max(times[:,:,3]) + 10)

        indices = [int(c[1]) - 1 for c in s]
        cells = [[setupLabels[i][0] for i in indices], [setupLabels[i][1] for i in indices]]
        labels = ["  r  ", "  p  "] if i == 0 else None
        
        plt.table(
            bbox=[0, -0.17, 1, 0.12],
            rowLabels=labels,
            cellText=cells
        )
    
    plt.savefig(os.path.join(os.path.join(outputDir, 'Grouped Cluster Configurations Task Times')))
    plt.clf()

    # Accuracy line plot
    for i in range(x):
        plt.plot(np.arange(1, y + 1), stats[i,:,1], label="r={}, p={}".format(*setupLabels[i]))
    
    plt.subplots_adjust(left=0.125, bottom=0.1)
    plt.legend()
    plt.xlim(1, y)
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Accuracy Line Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Accuracy Line Plot')))
    plt.clf()

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            plt.plot(np.arange(1, y + 1), stats[int(g[1]) - 1,:,1], label="r={}, p={}".format(*setupLabels[int(g[1]) - 1]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])

        plt.legend()
        plt.xlim(1, y)
        plt.xticks(np.arange(1, y + 1, hyperParameterStep))
        plt.yticks(np.arange(0, 101, 10))
        plt.ylim(0, 100)
        plt.xlabel('Hyper-parameter Set (#)')
        plt.ylabel('Accuracy (%)')
        plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Accuracy Line Plot (Group {})'.format(i + 1))))
        plt.clf()

    # Accuracy scatter plot
    for i in range(x):
        plt.scatter(np.arange(1, y + 1), stats[i,:,1], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.xlim(0, y + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Accuracy Scatter Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Accuracy Scatter Plot')))
    plt.clf()    

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            plt.scatter(np.arange(1, y + 1), stats[int(g[1]) - 1,:,1], label="r={}, p={}".format(*setupLabels[int(g[1]) - 1]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])

        plt.legend()
        plt.xticks(np.arange(1, y + 1, hyperParameterStep))
        plt.xlim(0, y + 1)
        plt.yticks(np.arange(0, 101, 10))
        plt.ylim(0, 100)
        plt.xlabel('Hyper-parameter Set (#)')
        plt.ylabel('Accuracy (%)')
        plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Accuracy Scatter Plot (Group {})'.format(i + 1))))
        plt.clf()

    # Job times line plot
    for i in range(x):
        plt.plot(np.arange(1, y + 1), times[i,:,1], label="r={}, p={}".format(*setupLabels[i]))
    
    plt.legend()
    plt.xlim(1, y)
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.yticks(np.arange(0, np.max(times[:,:,1]) + 10, jobTimeStep), jobTimeTicks)
    plt.ylim(0, np.max(times[:,:,1]) + 10)
    # plt.title('Job Times Line Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Job Times (sec)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Job Times Line Plot')))
    plt.clf()

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            plt.plot(np.arange(1, y + 1), times[int(g[1]) - 1,:,1], label="r={}, p={}".format(*setupLabels[int(g[1]) - 1]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])

        plt.legend()
        plt.xlim(1, y)
        plt.xticks(np.arange(1, y + 1, hyperParameterStep))
        plt.yticks(np.arange(0, np.max(times[:,:,1]) + 10, jobTimeStep), jobTimeTicks)
        plt.ylim(0, np.max(times[:,:,1]) + 10)
        plt.xlabel('Hyper-parameter Set (#)')
        plt.ylabel('Job Times (sec)')
        plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Job Times Line Plot (Group {})'.format(i + 1))))
        plt.clf()

    # Job times scatter plot
    for i in range(x):
        plt.scatter(np.arange(1, y + 1), times[i,:,1], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.xlim(0, y + 1)
    plt.yticks(np.arange(0, np.max(times[:,:,1]) + 10, jobTimeStep), jobTimeTicks)
    plt.ylim(0, np.max(times[:,:,1]) + 10)
    # plt.title('Job Times Scatter Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Job Times (sec)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Job Times Scatter Plot')))
    plt.clf()   

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            plt.scatter(np.arange(1, y + 1), times[int(g[1]) - 1,:,1], label="r={}, p={}".format(*setupLabels[int(g[1]) - 1]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])

        plt.legend()
        plt.xticks(np.arange(1, y + 1, hyperParameterStep))
        plt.xlim(0, y + 1)
        plt.yticks(np.arange(0, np.max(times[:,:,1]) + 10, jobTimeStep), jobTimeTicks)
        plt.ylim(0, np.max(times[:,:,1]) + 10)
        plt.xlabel('Hyper-parameter Set (#)')
        plt.ylabel('Job Times (sec)')
        plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Job Times Scatter Plot (Group {})'.format(i + 1))))
        plt.clf()

    # Task times line plot
    for i in range(x):
        plt.plot(np.arange(1, y + 1), times[i,:,3], label="r={}, p={}".format(*setupLabels[i]))
    
    plt.legend()
    plt.subplots_adjust(left=0.15)
    plt.xlim(1, y)
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.yticks(np.arange(0, np.max(times[:,:,3]) + 10, taskTimeStep), taskTimeTicks)
    plt.ylim(0, np.max(times[:,:,3]) + 10)
    # plt.title('Task Times Line Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Task Times (sec)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Task Times Line Plot')))
    plt.clf()

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            plt.plot(np.arange(1, y + 1), times[int(g[1]) - 1,:,3], label="r={}, p={}".format(*setupLabels[int(g[1]) - 1]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])

        plt.legend()
        plt.xlim(1, y)
        plt.xticks(np.arange(1, y + 1, hyperParameterStep))
        plt.yticks(np.arange(0, np.max(times[:,:,3]) + 10, taskTimeStep), taskTimeTicks)
        plt.ylim(0, np.max(times[:,:,3]) + 10)
        plt.xlabel('Hyper-parameter Set (#)')
        plt.ylabel('Task Times (sec)')
        plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Task Times Line Plot (Group {})'.format(i + 1))))
        plt.clf()

    # Task times scatter plot
    for i in range(x):
        plt.scatter(np.arange(1, y + 1), times[i,:,3], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.xticks(np.arange(1, y + 1, hyperParameterStep))
    plt.xlim(0, y + 1)
    plt.yticks(np.arange(0, np.max(times[:,:,3]) + 10, taskTimeStep), taskTimeTicks)
    plt.ylim(0, np.max(times[:,:,3]) + 10)
    # plt.title('Task Times Scatter Plot')
    plt.xlabel('Hyper-parameter Set (#)')
    plt.ylabel('Task Times (sec)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Task Times Scatter Plot')))
    plt.clf()   

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            plt.scatter(np.arange(1, y + 1), times[int(g[1]) - 1,:,3], label="r={}, p={}".format(*setupLabels[int(g[1]) - 1]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])

        plt.legend()
        plt.xticks(np.arange(1, y + 1, hyperParameterStep))
        plt.xlim(0, y + 1)
        plt.yticks(np.arange(0, np.max(times[:,:,3]) + 10, taskTimeStep), taskTimeTicks)
        plt.ylim(0, np.max(times[:,:,3]) + 10)
        plt.xlabel('Hyper-parameter Set (#)')
        plt.ylabel('Task Times (sec)')
        plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Task Times Scatter Plot (Group {})'.format(i + 1))))
        plt.clf()

    # Configuration accuracy with best HP marked
    bestHP = np.argmax(stats[0,:,1])

    for i in range(x):
        plt.scatter([i + 1] * y, stats[i,:,1], label="r={}, p={}".format(*setupLabels[i]))
        plt.scatter([i + 1], stats[i,bestHP,1], color="black")

    plt.legend()
    plt.xticks(np.arange(1, x + 1))
    plt.xlim(0, x + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    # plt.title('Configuration Accuracy Scatter Plot (best HP)')
    plt.xlabel('System Configuration (#)')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Accuracy Scatter Plot (best HP)')))
    plt.clf()  

    # Throughput and latency graphs
    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            plt.plot(np.arange(1, y + 1), throughputs[int(g[1]) - 1,:], label="r={}, p={}".format(*setupLabels[int(g[1]) - 1]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])

        plt.legend()
        plt.xlim(1, y)
        plt.xticks(np.arange(1, y + 1, hyperParameterStep))
        plt.yticks(np.arange(0, np.max(throughputs[:,:]) + 10, throughputStep), throughputTicks)
        plt.ylim(0, np.max(throughputs[:,:]) + 10)
        plt.xlabel('Hyper-parameter Set (#)')
        plt.ylabel('Throughput (x1000 recs/sec)')
        plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Throughput Line Plot (Group {})'.format(i + 1))))
        plt.clf()

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            plt.scatter(np.arange(1, y + 1), throughputs[int(g[1]) - 1,:], label="r={}, p={}".format(*setupLabels[int(g[1]) - 1]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])

        plt.legend()
        plt.xlim(1, y)
        plt.xticks(np.arange(1, y + 1, hyperParameterStep))
        plt.yticks(np.arange(0, np.max(throughputs[:,:]) + 10, throughputStep), throughputTicks)
        plt.ylim(0, np.max(throughputs[:,:]) + 10)
        plt.xlabel('Hyper-parameter Set (#)')
        plt.ylabel('Throughput (x1000 recs/sec)')
        plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Throughput Scatter Plot (Group {})'.format(i + 1))))
        plt.clf()

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            plt.plot(np.arange(1, y + 1), latencies[int(g[1]) - 1,:], label="r={}, p={}".format(*setupLabels[int(g[1]) - 1]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])

        plt.subplots_adjust(left=0.15)
        plt.legend()
        plt.xlim(1, y)
        plt.xticks(np.arange(1, y + 1, hyperParameterStep))
        plt.yticks(np.arange(0, np.max(latencies[:,:]) + 10, latencyStep), latencyTicks)
        plt.ylim(0, np.max(latencies[:,:]) + 10)
        plt.xlabel('Hyper-parameter Set (#)')
        plt.ylabel('Latency (ms)')
        plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Latency Line Plot (Group {})'.format(i + 1))))
        plt.clf()

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            plt.scatter(np.arange(1, y + 1), latencies[int(g[1]) - 1,:], label="r={}, p={}".format(*setupLabels[int(g[1]) - 1]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])

        plt.legend()
        plt.xlim(1, y)
        plt.xticks(np.arange(1, y + 1, hyperParameterStep))
        plt.yticks(np.arange(0, np.max(latencies[:,:]) + 10, latencyStep), latencyTicks)
        plt.ylim(0, np.max(latencies[:,:]) + 10)
        plt.xlabel('Hyper-parameter Set (#)')
        plt.ylabel('Latency (ms)')
        plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Latency Scatter Plot (Group {})'.format(i + 1))))
        plt.clf()

    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True)
    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System Configuration", labelpad=25)
    plt.ylabel("Throughput (x1000 recs/sec)", labelpad=10)
    plt.subplots_adjust(bottom=0.25)
        
    bestHP = np.argmax(throughputs[0,:])

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            axs[i].scatter([j + 1] * y, throughputs[int(g[1]) - 1,:], label=g, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])
            axs[i].scatter([j + 1], throughputs[int(g[1]) - 1,bestHP], color="black")

        plt.sca(axs[i])
        plt.xticks([])
        plt.xlim(0.5, 4.5)
        plt.yticks(np.arange(0, np.max(throughputs[:,:]) + 10, throughputStep), throughputTicks)
        plt.ylim(0, np.max(throughputs[:,:]) + 10)

        indices = [int(c[1]) - 1 for c in s]
        cells = [[setupLabels[i][0] for i in indices], [setupLabels[i][1] for i in indices]]
        labels = ["  r  ", "  p  "] if i == 0 else None

        plt.table(
            bbox=[0, -0.17, 1, 0.12],
            rowLabels=labels,
            cellText=cells
        )

    plt.savefig(os.path.join(os.path.join(outputDir, 'Grouped Cluster Configurations Throughput')))
    plt.clf()

    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System Configuration", labelpad=25)
    plt.ylabel("Latency (ms)", labelpad=20)
    plt.subplots_adjust(bottom=0.25)
        
    bestHP = np.argmin(latencies[0,:])

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            axs[i].scatter([j + 1] * y, latencies[int(g[1]) - 1,:], label=g, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])
            axs[i].scatter([j + 1], latencies[int(g[1]) - 1,bestHP], color="black")

        plt.sca(axs[i])
        plt.xticks([])
        plt.xlim(0.5, 4.5)
        plt.yticks(np.arange(0, np.max(latencies[:,:]) + 10, latencyStep), latencyTicks)
        plt.ylim(0, np.max(latencies[:,:]) + 10)

        indices = [int(c[1]) - 1 for c in s]
        cells = [[setupLabels[i][0] for i in indices], [setupLabels[i][1] for i in indices]]
        labels = ["  r  ", "  p  "] if i == 0 else None
        
        plt.table(
            bbox=[0, -0.17, 1, 0.12],
            rowLabels=labels,
            cellText=cells
        )
    
    plt.savefig(os.path.join(os.path.join(outputDir, 'Grouped Cluster Configurations Latency')))
    plt.clf()


    for i in range(x):
        plt.scatter(stats[i,:,1], throughputs[i,:], label="r={}, p={}".format(*setupLabels[i]))

    plt.subplots_adjust(left=0.125, bottom=0.11)
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(50, 101, 10))
    plt.xlim(50, 100)
    # plt.yticks(np.arange(0, np.max(throughputs) + 1000, 1000), np.arange(0, ceil((np.max(throughputs) + 1000) / 1000)))
    # plt.ylim(0, floor((np.max(throughputs) + 1000) / 1000) * 1000)
    # plt.yticks(np.arange(0, np.max(throughputs) + 1000, throughputStep), throughputTicks)
    # plt.ylim(0, ceil(np.max(throughputs) / 1000))

    top = ceil(np.max(throughputs) / 1000) * 1000
    step = max(top / 10, 1000)
    labels = np.arange(0, top + 1, step)
    ticks = np.arange(0, (top + 1) / 1000, step / 1000)
    
    # print(top, step, labels, ticks)

    plt.yticks(labels, ticks)
    plt.ylim(0, top)
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Throughput (x1000 recs/sec)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Accuracy Throughput Scatter Plot')))
    plt.clf()
    
    for i in range(x):
        plt.scatter(stats[i,:,1], latencies[i,:], label="r={}, p={}".format(*setupLabels[i]))

    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(50, 101, 10))
    plt.xlim(50, 100)
    plt.yticks(np.arange(0, np.max(latencies[:,:]) + 10, latencyStep), latencyTicks)
    plt.ylim(0, np.max(latencies[:,:]) + 10)
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Latency (ms)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Accuracy Latency Scatter Plot')))
    plt.clf()

    accThrScore = normalizedStats[:,:,1] / 2 + normalizedThroughputs / 2
    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System Configuration", labelpad=25)
    plt.ylabel("Accuracy/Throughput Score (%)")
    plt.subplots_adjust(bottom=0.25)
        
    bestHP = np.argmax(accThrScore[0])

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            axs[i].scatter([j + 1] * y, accThrScore[int(g[1]) - 1], label=g, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])
            axs[i].scatter([j + 1], accThrScore[int(g[1]) - 1, bestHP], color="black")

        plt.sca(axs[i])
        plt.xticks([])
        plt.xlim(0.5, 4.5)
        plt.yticks(np.arange(0, 101, 10))
        plt.ylim(0, 100)

        indices = [int(c[1]) - 1 for c in s]
        cells = [[setupLabels[i][0] for i in indices], [setupLabels[i][1] for i in indices]]
        labels = ["  r  ", "  p  "] if i == 0 else None

        plt.table(
            bbox=[0, -0.17, 1, 0.12],
            rowLabels=labels,
            cellText=cells
        )
    
    plt.savefig(os.path.join(os.path.join(outputDir, 'Grouped Cluster Configurations Accuracy-Throughput Score')))
    plt.clf()

    accLatScore = normalizedStats[:,:,1] / 2 + normalizedLatencies / 2
    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System Configuration", labelpad=25)
    plt.ylabel("Accuracy/Latency Score (%)")
    plt.subplots_adjust(bottom=0.25)
        
    bestHP = np.argmax(accLatScore[0])

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            axs[i].scatter([j + 1] * y, accLatScore[int(g[1]) - 1], label=g, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])
            axs[i].scatter([j + 1], accLatScore[int(g[1]) - 1, bestHP], color="black")

        plt.sca(axs[i])
        plt.xticks([])
        plt.xlim(0.5, 4.5)
        plt.yticks(np.arange(0, 101, 10))
        plt.ylim(0, 100)

        indices = [int(c[1]) - 1 for c in s]
        cells = [[setupLabels[i][0] for i in indices], [setupLabels[i][1] for i in indices]]
        labels = ["  r  ", "  p  "] if i == 0 else None

        plt.table(
            bbox=[0, -0.17, 1, 0.12],
            rowLabels=labels,
            cellText=cells
        )
    
    plt.savefig(os.path.join(os.path.join(outputDir, 'Grouped Cluster Configurations Accuracy-Latency Score')))
    plt.clf()

    accThrLatScore = normalizedStats[:,:,1] / 3 + normalizedThroughputs / 3 + normalizedLatencies / 3
    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System Configuration", labelpad=25)
    plt.ylabel("Accuracy/Throughput/Latency Score (%)")
    plt.subplots_adjust(bottom=0.25)
        
    bestHP = np.argmax(accThrLatScore[0])

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            axs[i].scatter([j + 1] * y, accThrLatScore[int(g[1]) - 1], label=g, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])
            axs[i].scatter([j + 1], accThrLatScore[int(g[1]) - 1, bestHP], color="black")

        plt.sca(axs[i])
        plt.xticks([])
        plt.xlim(0.5, 4.5)
        plt.yticks(np.arange(0, 101, 10))
        plt.ylim(0, 100)

        indices = [int(c[1]) - 1 for c in s]
        cells = [[setupLabels[i][0] for i in indices], [setupLabels[i][1] for i in indices]]
        labels = ["  r  ", "  p  "] if i == 0 else None

        plt.table(
            bbox=[0, -0.17, 1, 0.12],
            rowLabels=labels,
            cellText=cells
        )
    
    plt.savefig(os.path.join(os.path.join(outputDir, 'Grouped Cluster Configurations Accuracy-Throughput-Latency Score')))
    plt.clf()

    accThrScore = np.sqrt((1 - normalizedStats[:,:,1] / 100)**2 + (1 - normalizedThroughputs / 100)**2)
    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System Configuration", labelpad=25)
    plt.ylabel("Accuracy/Throughput Pareto Distance (%)")
    plt.subplots_adjust(bottom=0.25)
        
    bestHP = np.argmin(accThrScore[0])

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            axs[i].scatter([j + 1] * y, accThrScore[int(g[1]) - 1] * 100, label=g, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])
            axs[i].scatter([j + 1], accThrScore[int(g[1]) - 1, bestHP] * 100, color="black")

        plt.sca(axs[i])
        plt.xticks([])
        plt.xlim(0.5, 4.5)
        plt.yticks(np.arange(0, 201, 20))
        plt.ylim(0, 200)

        indices = [int(c[1]) - 1 for c in s]
        cells = [[setupLabels[i][0] for i in indices], [setupLabels[i][1] for i in indices]]
        labels = ["  r  ", "  p  "] if i == 0 else None

        plt.table(
            bbox=[0, -0.17, 1, 0.12],
            rowLabels=labels,
            cellText=cells
        )
    
    plt.savefig(os.path.join(os.path.join(outputDir, 'Grouped Cluster Configurations Accuracy-Throughput Pareto Distance')))
    plt.clf()

    accLatScore = np.sqrt((1 - normalizedStats[:,:,1] / 100)**2 + (1 - normalizedLatencies / 100)**2)    
    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System Configuration", labelpad=25)
    plt.ylabel("Accuracy/Latency Pareto Distance (%)")
    plt.subplots_adjust(bottom=0.25)
        
    bestHP = np.argmin(accLatScore[0])

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            axs[i].scatter([j + 1] * y, accLatScore[int(g[1]) - 1] * 100, label=g, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])
            axs[i].scatter([j + 1], accLatScore[int(g[1]) - 1, bestHP] * 100, color="black")

        plt.sca(axs[i])
        plt.xticks([])
        plt.xlim(0.5, 4.5)
        plt.yticks(np.arange(0, 201, 20))
        plt.ylim(0, 200)

        indices = [int(c[1]) - 1 for c in s]
        cells = [[setupLabels[i][0] for i in indices], [setupLabels[i][1] for i in indices]]
        labels = ["  r  ", "  p  "] if i == 0 else None

        plt.table(
            bbox=[0, -0.17, 1, 0.12],
            rowLabels=labels,
            cellText=cells
        )
    
    plt.savefig(os.path.join(os.path.join(outputDir, 'Grouped Cluster Configurations Accuracy-Latency Pareto Distance')))
    plt.clf()

    accThrLatScore = np.sqrt((1 - normalizedStats[:,:,1] / 100)**2 + (1 - normalizedThroughputs / 100)**2 + (1 - normalizedLatencies / 100)**2)
    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System Configuration", labelpad=25)
    plt.ylabel("Accuracy/Throughput/Latency Pareto Distance (%)")
    plt.subplots_adjust(bottom=0.25)
        
    bestHP = np.argmin(accThrLatScore[0])

    for i,s in enumerate(setupGroups):
        for j,g in enumerate(s):
            axs[i].scatter([j + 1] * y, accThrLatScore[int(g[1]) - 1] * 100, label=g, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][int(g[1]) - 1])
            axs[i].scatter([j + 1], accThrLatScore[int(g[1]) - 1, bestHP] * 100, color="black")

        plt.sca(axs[i])
        plt.xticks([])
        plt.xlim(0.5, 4.5)
        plt.yticks(np.arange(0, 201, 20))
        plt.ylim(0, 200)

        indices = [int(c[1]) - 1 for c in s]
        cells = [[setupLabels[i][0] for i in indices], [setupLabels[i][1] for i in indices]]
        labels = ["  r  ", "  p  "] if i == 0 else None

        plt.table(
            bbox=[0, -0.17, 1, 0.12],
            rowLabels=labels,
            cellText=cells
        )
    
    plt.savefig(os.path.join(os.path.join(outputDir, 'Grouped Cluster Configurations Accuracy-Throughput-Latency Pareto Distance')))
    plt.clf()


def main(args):
    model = args[0]
    inputDir = args[1]
    outputDir = args[2]
    experiments = os.listdir(inputDir)
    experiments.sort()
    dataset = "agrawal" if "agrawal" in inputDir else None
    dataset = "tweets" if "tweets" in inputDir else None

    hps = list(fileIterator(model, dataset))

    # 1st dimension is experiments (i.e. s1, s2, ..., sn) and 2nd dimension is hyper-parameters
    stats = np.zeros((len(experiments), len(hps), 5))
    # The 3rd dimension is: instances, accuracy, recall, precision, f1Score
    times = np.zeros((len(experiments), len(hps), 4))
    # The 3rd dimension is: jobs, jobTime, tasks, taskTime (in ms)
    latencies = np.zeros((len(experiments), len(hps)))
    throughputs = np.zeros((len(experiments), len(hps)))

    for i, exp in enumerate(experiments):
        print("Processing experiment s{}/s{}".format(i + 1, len(experiments)))
        for j, fileName in enumerate(hps):
            
            # if j == 3:
            #     break

            print("\tProcessing file {} ({}/{})".format(fileName, j + 1, len(hps)))
            data = getData(os.path.join(inputDir, exp, fileName + '.csv'))
            stats[i,j,:] = data
            
            runSparkLogParser(os.path.join(inputDir, exp, fileName), os.path.join(outputDir, 'sparkLogs.txt'))
            jobs, jobTime, tasks, taskTime = parseSparkLogParser(os.path.join(outputDir, 'sparkLogs.txt'))
            times[i,j,:] = jobs, jobTime, tasks, taskTime

            throughputs[i,j] = 1000 * stats[i,j,0] / jobTime
            latencies[i,j] = 2 * taskTime / tasks
        
    print("Done processing.")
    os.remove(os.path.join(outputDir, 'sparkLogs.txt'))
    
    print("Normalizing values....")
    normalizedStats = np.copy(stats)
    for i in range(1, np.shape(normalizedStats)[2]):
        normalizedStats[:,:,i] = normalize(normalizedStats[:,:,i], np.min(normalizedStats[:,:,i]), np.max(normalizedStats[:,:,i]))

    normalizedTimes = np.copy(times)
    for i in range(1, np.shape(normalizedTimes)[2], 2):
        normalizedTimes[:,:,i] = inverseNormalize(normalizedTimes[:,:,i], np.min(normalizedTimes[:,:,i]), np.max(normalizedTimes[:,:,i]))

    normalizedThroughputs = normalize(throughputs, np.min(throughputs), np.max(throughputs))
    
    normalizedLatencies = inverseNormalize(latencies, np.min(latencies), np.max(latencies))

    print("Writting stats to CSV file...")
    writeStatsToCSVFile(outputDir, hps, stats, normalizedStats, times, normalizedTimes)

    print("Generating graphs...")
    generateGraphs(outputDir, hps, stats, normalizedStats, times, normalizedTimes, throughputs, normalizedThroughputs, latencies, normalizedLatencies)
    
    print("Done.")

if __name__ == '__main__':
    if len(sys.argv[1:]) != 3:
        print("Usage: <model> <inputDir> <outputDir>")
        print("\t<model> A string specifying which model is used (i.e. 'HT', 'SGD')")
        print("\t<inputDir> contains subdirectories of experiments. (i.e. s1, s2, ..., sn)")
        print("\t<outputDir> is the directory you wish to store the generated results and graphs.")
        exit(0)

    main(sys.argv[1:])