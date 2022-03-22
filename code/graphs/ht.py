import os
import sys
import csv
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def fileIterator():
    prefix = 'spark_ht'
    for c in ['0.01', '0.1']:
        for t in ['0.05', '0.1']:
            for g in ['200', '500']:
                for s in ['InfoGainSplitCriterion', 'GiniSplitCriterion']:
                    for d in ['20', '30']:
                        yield "{}_c{}_t{}_g{}_s{}_d{}".format(prefix, c, t, g, s, d)

def readConfs(inputDirs):
    confs = []
    for inputDir in inputDirs:
        conf = {}
        confFile = open(os.path.join(inputDir, 'spark-defaults.conf'), 'r')
        for line in confFile:
            if line[0] != '#' and line[0] != '\n':
                k,v = line.split()
                conf[k] = v

        confFile.close()
        confs.append(conf)
    return confs

def getData(fileName):
    instance = 0
    instances = []
    accuracy = []
    recall = []
    precission = []
    f1Score = []

    fp = open(fileName, 'r')
    next(fp)
    
    for line in fp:
        i = int(float(line.split(',')[1]))
        instance += i
        instances.append(instance)
        accuracy.append(float(line.split(',')[2]) * 100)
        recall.append(float(line.split(',')[3]) * 100)
        precission.append(float(line.split(',')[4]) * 100)
        f1Score.append(float(line.split(',')[5]) * 100)

    fp.close()
    return instances, accuracy, recall, precission, f1Score

def plotConfHeader(confs, ax):
    conf = confs[0]
    header = '\n'.join(conf.keys())
    ax.text(0, 0, header)
    ax.axis('off')

def plotConf(conf, ax):
    values = '\n'.join(conf.values())
    ax.text(0, 0, values)
    ax.axis('off')

def plotData(fileName, data, ax):
    ax.plot(data[0], data[1], label="Accuracy")
    ax.plot(data[0], data[2], label="Recall")
    ax.plot(data[0], data[3], label="Precission")
    ax.plot(data[0], data[4], label="F1-Score")
    # ax.set(xlabel='Instance (#)', ylabel='Performance (%)') #, title='.'.join(os.path.basename(fileName).split('.')[:-1]))
    # ax.set_title('Avg accuracy (last {}k instances): {:.2f}%'.format(int(np.floor(len(data[0])/2) * data[0][0] / 1000), avgAccuracyLastHalf(data[1])))
    ax.set_title('Avg accuracy (last {}k instances): {:.2f}%'.format(int(np.floor(data[0][-1] / 2000)), avgAccuracyLastHalf(data[1])))
    ax.set_xlim([0, data[0][-1]])
    ax.set_xticks(np.arange(0, data[0][-1] + 1, data[0][-1] / 5))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid()
    ax.legend()

def getHTDepth(fileName):
    depth = 0
    fp = open(fileName, 'r')
    for line in fp.readlines():
        if 'depth' in line:
            depth = max(depth, int(line.split('depth: ')[1].split(' |')[0]))

    fp.close()
    return depth
    
def avgAccuracyLastHalf(data):
    return np.average(data[int(np.floor(len(data)/2)):])

def writeStatsToFile(outputDir, confs, hps, results, overallAccuracies, jobTimes, taskTimes):
    matrix = np.array(results)
  
    best_accuracy = np.max(matrix)
    best_confs = np.argmax(matrix)

    best_SP_per_HP = np.argmax(matrix, 1)
    best_accuracies_per_HP = np.max(matrix, 1)

    best_HP_per_SP = np.argmax(matrix, 0)
    best_accuracies_per_SP = np.max(matrix, 0)

    matrix = np.array(overallAccuracies)

    overall_best_confs = np.argmax(matrix)
    overall_best_accuracy = np.max(matrix)

    overall_best_SP_per_HP = np.argmax(matrix, 1)
    overall_best_accuracies_per_HP = np.max(matrix, 1)

    overall_best_HP_per_SP = np.argmax(matrix, 0)
    overall_best_accuracies_per_SP = np.max(matrix, 0)

    fp = open(os.path.join(outputDir, 'results.txt'), 'w')
    fp.write('Results\n')
    fp.write('=======\n')
    fp.write('\n')
    for i,conf in enumerate(confs):
        fp.write('Configuration {}\n'.format(i + 1))
        fp.write('=' * len('Configuration {}'.format(i)) + '\n')
        for k,v in conf.items():
            fp.write('{:41} {:>60}\n'.format(k,v))
        fp.write('\n')

    fp.write('Average Job and Task times per configuration\n')
    fp.write('============================================\n')

    fp.write('{:60}|{:>20}|{:>20}\n'.format('', 'Avg. Job Time (ms)', 'Avg. Task Time (ms)'))
    fp.write('{:60}'.format('Hyper Parameters'))
    for conf in range(len(confs)):
        fp.write('|{:>6}'.format(conf + 1))
    for conf in range(len(confs)):
        fp.write('|{:>6}'.format(conf + 1))
    fp.write('\n')
    fp.write('=' * 102 + '\n')
    
    for i, h in enumerate(hps):
        fp.write('{:60}'.format(h))
        for j in jobTimes[i]:
            fp.write('|{:>6}'.format(j))
        for t in taskTimes[i]:
            fp.write('|{:>6}'.format(t))
        fp.write('\n')
    fp.write('\n')


    fp.write('{:60}|{:>20}|{:>20}\n'.format('Hyper Parameters', 'Best Configuration', 'Accuracy (%)'))
    fp.write('=' * 102)
    fp.write('\n')

    for i,row in enumerate(best_SP_per_HP):
        fp.write('{:60}|{:>20}|{:>20.2f}\n'.format(hps[i], row + 1, best_accuracies_per_HP[i]))

    fp.write('{:60}|{:>20}|{:>20}\n'.format('Hyper Parameters', 'Best Configuration', 'Overall Accuracy (%)'))
    fp.write('=' * 102)
    fp.write('\n')

    for i,row in enumerate(overall_best_SP_per_HP):
        fp.write('{:60}|{:>20}|{:>20.2f}\n'.format(hps[i], row + 1, overall_best_accuracies_per_HP[i]))
    
    fp.write('\n')

    fp.write('{:20}|{:>60}|{:>20}\n'.format('Configuration', 'Best Hyper Parameters', 'Accuracy (%)'))
    fp.write('=' * 102)
    fp.write('\n')

    for i, row in enumerate(best_HP_per_SP):
        fp.write('{:20}|{:>60}|{:>20.2f}\n'.format(i + 1, hps[row], best_accuracies_per_SP[i]))

    fp.write('\n')

    fp.write('{:20}|{:>60}|{:>20}\n'.format('Configuration', 'Best Hyper Parameters', 'Overall Accuracy (%)'))
    fp.write('=' * 102)
    fp.write('\n')

    for i, row in enumerate(overall_best_HP_per_SP):
        fp.write('{:20}|{:>60}|{:>20.2f}\n'.format(i + 1, hps[row], overall_best_accuracies_per_SP[i]))

    fp.write('\n')

    fp.write('{:20}|{:>60}|{:>20}\n'.format('Best Configuration', 'Best Hyper Parameters', 'Accuracy (%)'))
    fp.write('=' * 102)
    fp.write('\n')
    fp.write('{:20}|{:>60}|{:>19.2f}%\n'.format(best_confs % len(confs) + 1, hps[int(best_confs / len(confs))], best_accuracy))
    fp.write('\n')

    fp.write('{:20}|{:>60}|{:>20}\n'.format('Best Configuration', 'Best Hyper Parameters', 'Overall Accuracy (%)'))
    fp.write('=' * 102)
    fp.write('\n')
    fp.write('{:20}|{:>60}|{:>19.2f}%\n'.format(overall_best_confs % len(confs) + 1, hps[int(overall_best_confs / len(confs))], overall_best_accuracy))
    fp.write('\n')

    fp.close()

def writeHTDepthToFile(outputDir, confs, hps, htDepth):
    fp = open(os.path.join(outputDir, 'results.txt'), 'a+')

    fp.write('HT Depths\n')
    fp.write('=========\n')

    fp.write('{:60}|{:>23}\n'.format('', 'Depth/Configuration'))
    fp.write('{:60}'.format('Hyper Parameters'))
    for conf in range(len(confs)):
        fp.write('|{:>7}'.format(conf + 1))
    fp.write('\n')
    fp.write('=' * 84 + '\n')

    for i,depths in enumerate(htDepth):
        fp.write('{:60}'.format(hps[i]))
        for j, depth in enumerate(depths):
            fp.write('|{:>7}'.format(depth))
        fp.write('\n')

    fp.write('\n')
    fp.close()

def hpsValues(hps):
    r = hps.split("_")[2:]
    return [x[1:] for x in r]

def normalize(value, min, max):
    return (value - min) / (max - min) * 100

def inverseNormalize(value, min, max):
    return (max - value) / (max - min) * 100

def writeStatsToCSVFile(outputDir, confs, hps, results, overallAccuracies, f1Scores, jobTimes, taskTimes, htDepths):
    accuracies = np.array(results)
    normalizedAccuracies = normalize(accuracies, np.min(accuracies), np.max(accuracies))

    overallAccuracies = np.array(overallAccuracies)
    normalizedOverallAccuracies = normalize(overallAccuracies, np.min(overallAccuracies), np.max(overallAccuracies))

    f1Scores = np.array(f1Scores)
    normalizedF1Scores = normalize(f1Scores, np.min(f1Scores), np.max(f1Scores))
    
    jobTimes = np.array(jobTimes)
    normalizedJobtimes = inverseNormalize(jobTimes, np.min(jobTimes), np.max(jobTimes))

    taskTimes = np.array(taskTimes)
    normalizedTaskTimes = inverseNormalize(taskTimes, np.min(taskTimes), np.max(taskTimes))

    with open(os.path.join(outputDir, "results.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["c", "t", "g", "s", "d", "Configuration", "HT Depth", "Accuracy (%)", "Normalized Accuracy (%)", "Overall Accuracy (%)", "Normalized Overall Accuracy (%)", "F1-Score (%)", "Normalized F1-Score (%)", "Average Job Time (sec)", "Normalized Average Job Time (%)", "Average Task Time (sec)", "Normalized Average Task Time (%)", "Average Normalized Time Joined (%)"])
        for i, r in enumerate(results):
            for j, c in enumerate(r):
                writer.writerow(hpsValues(hps[i]) + [j + 1, htDepths[i][j], accuracies[i][j], normalizedAccuracies[i][j], overallAccuracies[i][j], normalizedOverallAccuracies[i][j], f1Scores[i][j], normalizedF1Scores[i][j], jobTimes[i][j], normalizedJobtimes[i][j], taskTimes[i][j], normalizedTaskTimes[i][j], (normalizedJobtimes[i][j] + normalizedTaskTimes[i][j]) / 2]) 

def linePlot(outputDir, accuracies, overallAccuracies):
    accuracies = np.array(accuracies)
    normalizedAccuracies = normalize(accuracies, np.min(accuracies), np.max(accuracies)) 

    overallAccuracies = np.array(overallAccuracies)
    normalizedOverallAccuracies = normalize(overallAccuracies, np.min(overallAccuracies), np.max(overallAccuracies)) 

    cases, confs = np.shape(normalizedAccuracies)

    for conf in range(confs):
        plt.plot(np.arange(1, cases + 1), normalizedAccuracies[:,conf], label="Configuration {}".format(conf + 1))

    plt.legend()
    plt.xlim(1, cases)
    plt.xticks(np.arange(1, cases + 1, 10))
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Line Plot')
    plt.xlabel('Case')
    plt.ylabel('Normalized Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Line Plot')))
    plt.clf()

    for conf in range(confs):
        plt.plot(np.arange(1, cases + 1), normalizedOverallAccuracies[:,conf], label="Configuration {}".format(conf + 1))

    plt.legend()
    plt.xlim(1, cases)
    plt.xticks(np.arange(1, cases + 1, 10))
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Overall Accuracy Line Plot')
    plt.xlabel('Case')
    plt.ylabel('Normalized Overall Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Overall Accuracy Line Plot')))
    plt.clf()

def scatterPlots(outputDir, accuracies, overallAccuracies, jobTimes, taskTimes):
    accuracies = np.array(accuracies)
    normalizedAccuracies = normalize(accuracies, np.min(accuracies), np.max(accuracies))
    
    overallAccuracies = np.array(overallAccuracies)
    normalizedOverallAccuracies = normalize(overallAccuracies, np.min(overallAccuracies), np.max(overallAccuracies))

    jobTimes = np.array(jobTimes)
    normalizedJobtimes = inverseNormalize(jobTimes, np.min(jobTimes), np.max(jobTimes))

    taskTimes = np.array(taskTimes)
    normalizedTaskTimes = inverseNormalize(taskTimes, np.min(taskTimes), np.max(taskTimes))

    normalizedJoinedTimes = (normalizedJobtimes + normalizedTaskTimes) / 2

    cases, confs = np.shape(normalizedJoinedTimes)

    for conf in range(confs):
        plt.scatter(normalizedAccuracies[:,conf], normalizedJoinedTimes[:,conf], label="Configuration {}".format(conf + 1))

    plt.legend()
    plt.xticks(np.arange(0, 101, 10))
    plt.xlim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Scatter Plot')
    plt.xlabel('Normalized Accuracy (%)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Scatter Plot')))
    plt.clf()

    for conf in range(confs):
        plt.scatter(normalizedOverallAccuracies[:,conf], normalizedJoinedTimes[:,conf], label="Configuration {}".format(conf + 1))

    plt.legend()
    plt.xticks(np.arange(0, 101, 10))
    plt.xlim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Overall Accuracy Scatter Plot')
    plt.xlabel('Normalized Overall Accuracy (%)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Overall Accuracy Scatter Plot')))
    plt.clf()

    for conf in range(confs):
        plt.scatter(np.arange(1, cases + 1), normalizedJoinedTimes[:,conf], label="Configuration {}".format(conf + 1))

    plt.legend()
    plt.xticks(np.arange(1, cases + 1, 1))
    plt.xlim(0, cases + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Experiment Times Scatter Plot')
    plt.xlabel('Experiment (#)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Times Scatter Plot')))
    plt.clf()

    for conf in range(confs):
        plt.scatter(np.arange(1, cases + 1), normalizedAccuracies[:,conf], label="Configuration {}".format(conf + 1))

    plt.legend()
    plt.xticks(np.arange(1, cases + 1, 1))
    plt.xlim(0, cases + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Experiment Accuracy Scatter Plot')
    plt.xlabel('Experiment (#)')
    plt.ylabel('Normalized Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Accuracy Scatter Plot')))
    plt.clf()

    for conf in range(confs):
        plt.scatter(np.arange(1, cases + 1), normalizedOverallAccuracies[:,conf], label="Configuration {}".format(conf + 1))

    plt.legend()
    plt.xticks(np.arange(1, cases + 1, 1))
    plt.xlim(0, cases + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Experiment Overall Accuracy Scatter Plot')
    plt.xlabel('Experiment (#)')
    plt.ylabel('Normalized Overall Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Overall Accuracy Scatter Plot')))
    plt.clf()

    for conf in range(confs):
        plt.scatter([conf + 1] * cases, normalizedJoinedTimes[:,conf], label="Configuration {}".format(conf + 1))

    plt.legend()
    plt.xticks(np.arange(1, confs + 1))
    plt.xlim(0, confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Configuration Times Scatter Plot')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Times Scatter Plot')))
    plt.clf()

    for conf in range(confs):
        plt.scatter([conf + 1] * cases, normalizedAccuracies[:,conf], label="Configuration {}".format(conf + 1))

    plt.legend()
    plt.xticks(np.arange(1, confs + 1))
    plt.xlim(0, confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Configuration Accuracy Scatter Plot')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Normalized Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Accuracy Scatter Plot')))
    plt.clf()

    for conf in range(confs):
        plt.scatter([conf + 1] * cases, normalizedOverallAccuracies[:,conf], label="Configuration {}".format(conf + 1))

    plt.legend()
    plt.xticks(np.arange(1, confs + 1))
    plt.xlim(0, confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Configuration Overall Accuracy Scatter Plot')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Normalized Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Overall Accuracy Scatter Plot')))
    plt.clf()

def whiskerPlots(outputDir, accuracies, overallAccuracies, jobTimes, taskTimes):
    accuracies = np.array(accuracies)
    normalizedAccuracies = normalize(accuracies, np.min(accuracies), np.max(accuracies))

    overallAccuracies = np.array(overallAccuracies)
    normalizedOverallAccuracies = normalize(overallAccuracies, np.min(overallAccuracies), np.max(overallAccuracies))
    
    jobTimes = np.array(jobTimes)
    normalizedJobtimes = inverseNormalize(jobTimes, np.min(jobTimes), np.max(jobTimes))

    taskTimes = np.array(taskTimes)
    normalizedTaskTimes = inverseNormalize(taskTimes, np.min(taskTimes), np.max(taskTimes))

    normalizedJoinedTimes = (normalizedJobtimes + normalizedTaskTimes) / 2

    cases, confs = np.shape(normalizedJoinedTimes)

    plt.boxplot([normalizedAccuracies[:,conf] for conf in range(confs)])
    plt.xticks(np.arange(1, confs + 1))
    plt.xlim(0, confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Whisker Plot (Normalized Accuracy).png')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Normalized Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (Normalized Accuracy)')))
    plt.clf()

    plt.boxplot([normalizedOverallAccuracies[:,conf] for conf in range(confs)])
    plt.xticks(np.arange(1, confs + 1))
    plt.xlim(0, confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Whisker Plot (Normalized Overall Accuracy).png')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Normalized Overall Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (Normalized Overall Accuracy)')))
    plt.clf()

    plt.boxplot([normalizedJoinedTimes[:,conf] for conf in range(confs)])
    plt.xticks(np.arange(1, confs + 1))
    plt.xlim(0, confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Whisker Plot (Normalized Times).png')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Normalized Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (Normalized Times)')))
    plt.clf()

    plt.boxplot([normalizedAccuracies[:,conf] / 3 + normalizedJobtimes[:,conf] / 3 + normalizedTaskTimes[:,conf] / 3 for conf in range(confs)])
    plt.xticks(np.arange(1, confs + 1))
    plt.xlim(0, confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Whisker Plot (Equally Weighted)')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Score (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (Equally Weighted)')))
    plt.clf()

    plt.boxplot([normalizedOverallAccuracies[:,conf] / 3 + normalizedJobtimes[:,conf] / 3 + normalizedTaskTimes[:,conf] / 3 for conf in range(confs)])
    plt.xticks(np.arange(1, confs + 1))
    plt.xlim(0, confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Whisker Plot (Equally Weighted - Normalized Overall Accuracy)')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Score (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (Equally Weighted - Normalized Overall Accuracy)')))
    plt.clf()

    plt.boxplot([normalizedAccuracies[:,conf] / 2 + normalizedJoinedTimes[:,conf] / 2 for conf in range(confs)])
    plt.xticks(np.arange(1, confs + 1))
    plt.xlim(0, confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Whisker Plot (50% Accuracy, 50% Times Joined)')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Score (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (50% Accuracy, 50% Times Joined)')))
    plt.clf()

    plt.boxplot([normalizedOverallAccuracies[:,conf] / 2 + normalizedJoinedTimes[:,conf] / 2 for conf in range(confs)])
    plt.xticks(np.arange(1, confs + 1))
    plt.xlim(0, confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Whisker Plot (50% Overall Accuracy, 50% Times Joined)')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Score (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Whisker Plot (50% Overall Accuracy, 50% Times Joined)')))
    plt.clf()

def runSparkLogParser(fileName, outputFile):
    # subprocess.run(['python2', '/home1/hadoop/streamingaggression/code/Spark-Log-Parser/main.py', fileName, outputFile])
    subprocess.run(['python2', '/home/lamody/streamingaggression/code/Spark-Log-Parser/main.py', fileName, outputFile])

def parseSparkLogParser(inputFile):
    jobs = []
    tasks = None

    job = False
    task = False

    fp = open(inputFile, 'r')
    for line in fp:
        if not job and '---> Jobs <---' in line:
            job = True
        
        if job:
            if 'Run time:' in line:
                time = line.split(': ')[1]
                time = int(time.strip()[:-2])
                jobs.append(time)

        if not task and '---> Tasks <---' in line:
            job = False
            task = True

        if task:
            if 'Task average runtime:' in line:
                time = '{:.2f}'.format(float(line.split(': ')[1].split()[0]))
                tasks = time
                break

    fp.close()
    return sum(jobs) / len(jobs), float(tasks)

def main(args):
    if len(args) < 2:
        print('Usage: input_dirs output_dir')
        sys.exit()
    
    inputDirs = args[:-1]
    outputDir = args[-1]

    confs = readConfs(inputDirs)
    fi =fileIterator()
    hps = []
    accuracies = []
    accuraciesLastHalf = []
    f1Scores = []
    htDepths = []  
    jobTimes = []
    taskTimes = []  
    for i, fileName in enumerate(fi):
        # if i == 2:
        #     break
        hps.append(fileName)

        print("Processing file: {}".format(fileName))
        fig, axs = plt.subplots(2, len(inputDirs) + 1, figsize=(15,7))
        fig.tight_layout()
        overallAccuracies = []
        avgAccuracies = []
        avgF1Scores = []
        depth = []
        jobs = []
        tasks = []
        for j, inputDir in enumerate(inputDirs):
            filePath = os.path.join(inputDir, fileName + '.csv')
            data = getData(filePath)
            overallAccuracies.append(np.average(data[1]))
            avgAccuracies.append(avgAccuracyLastHalf(data[1]))
            avgF1Scores.append(np.average(data[4]))
            plotConf(confs[j], axs[0, j + 1])
            plotData(filePath, data, axs[1, j + 1])
            depth.append(getHTDepth(os.path.join(inputDir, fileName + '.log')))
            runSparkLogParser(os.path.join(inputDir, fileName), os.path.join(outputDir, 'sparkLogs.txt'))
            j, t = parseSparkLogParser(os.path.join(outputDir, 'sparkLogs.txt'))
            jobs.append(j)
            tasks.append(t)
        
        accuracies.append(overallAccuracies)
        accuraciesLastHalf.append(avgAccuracies)
        f1Scores.append(avgF1Scores)
        jobTimes.append(jobs)
        taskTimes.append(tasks)
        htDepths.append(depth)
        
        plotConfHeader(confs, axs[0][0])
        axs[1,0].axis('off')
        fig.savefig(os.path.join(outputDir, fileName + '.png'))
        plt.close(fig)

    os.remove(os.path.join(os.path.join(outputDir, 'sparkLogs.txt')))
    print("Writting stats to TXT file...")
    writeStatsToFile(outputDir, confs, hps, accuraciesLastHalf, accuracies, jobTimes, taskTimes)
    writeHTDepthToFile(outputDir, confs, hps, htDepths)
    print("Writting stats to CSV file...")
    writeStatsToCSVFile(outputDir, confs, hps, accuraciesLastHalf, accuracies, f1Scores, jobTimes, taskTimes, htDepths)
    print("Generating line plot...")
    linePlot(outputDir, accuraciesLastHalf, accuracies)
    print("Generating scatter plots...")
    scatterPlots(outputDir, accuraciesLastHalf, accuracies, jobTimes, taskTimes)
    print("Generating whisker plots...")
    whiskerPlots(outputDir, accuraciesLastHalf, accuracies, jobTimes, taskTimes)
    print("Done.")
    
if __name__ == '__main__':
    main(sys.argv[1:])