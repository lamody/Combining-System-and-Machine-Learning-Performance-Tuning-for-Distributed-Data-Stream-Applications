import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

def normalize(value, min, max):
    return (value - min) / (max - min) * 100

def inverseNormalize(value, min, max):
    return (max - value) / (max - min) * 100

def getData(results, conf, accuracy, overallAccuracy, jobTime, taskTime):
    accuracies = []
    overallAccuracies = []
    jobTimes = []
    taskTimes = []
    c = 0
    
    print("Getting data...")
    for i,r in enumerate(results):
        fileName = os.path.join(r, "results.csv")
        acc = []
        overall_acc = []
        job = []
        task = []
        print("File: {} ({}/{})".format(fileName, i + 1, len(results)))
        with open(fileName, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for line, row in enumerate(reader):
                if line == 0:
                    continue

                c = max(c, int(row[conf]))

                acc.append(float(row[accuracy]))
                overall_acc.append(float(row[overallAccuracy]))
                job.append(float(row[jobTime]))
                task.append(float(row[taskTime]))

        accuracies.append(np.reshape(np.array(acc), (-1, c)))
        overallAccuracies.append(np.reshape(np.array(overall_acc), (-1, c)))
        jobTimes.append(np.reshape(np.array(job), (-1, c)))
        taskTimes.append(np.reshape(np.array(task), (-1, c)))

    return np.array(accuracies), np.array(overallAccuracies), np.array(jobTimes), np.array(taskTimes)  

def scatterPlots(accuracies, overallAccuracies, jobTimes, taskTimes, labels, outputDir):
    normalizedAccuracies = normalize(accuracies, np.min(accuracies), np.max(accuracies))
    normalizedOverallAccuracies = normalize(overallAccuracies, np.min(overallAccuracies), np.max(overallAccuracies))
    normalizedJobtimes = inverseNormalize(jobTimes, np.min(jobTimes), np.max(jobTimes))
    normalizedTaskTimes = inverseNormalize(taskTimes, np.min(taskTimes), np.max(taskTimes))
    normalizedJoinedTimes = (normalizedJobtimes + normalizedTaskTimes) / 2

    runs, cases, confs = np.shape(normalizedJoinedTimes)

    for i in range(runs):
        for j in range(confs):
            index = i * confs + j + 1
            plt.scatter([index] * cases, normalizedJoinedTimes[i,:,j], label="{} - Configuration {}".format(labels[i], j + 1))

    plt.legend()
    plt.xticks(np.arange(1, runs * confs + 1))
    plt.xlim(0, runs * confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Configuration Times Scatter Plot')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Times Scatter Plot')))
    plt.clf()

    for i in range(runs):
        for j in range(confs):
            index = i * confs + j + 1
            plt.scatter([index] * cases, normalizedAccuracies[i,:,j], label="{} - Configuration {}".format(labels[i], j + 1))

    plt.legend()
    plt.xticks(np.arange(1, runs * confs + 1))
    plt.xlim(0, runs * confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Configuration Accuracy Scatter Plot')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Normalized Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Accuracy Scatter Plot')))
    plt.clf()

    for i in range(runs):
        for j in range(confs):
            index = i * confs + j + 1
            plt.scatter([index] * cases, normalizedOverallAccuracies[i,:,j], label="{} - Configuration {}".format(labels[i], j + 1))

    plt.legend()
    plt.xticks(np.arange(1, runs * confs + 1))
    plt.xlim(0, runs * confs + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Configuration Overall Accuracy Scatter Plot')
    plt.xlabel('Configuration (#)')
    plt.ylabel('Normalized Overall Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Configuration Overall Accuracy Scatter Plot')))
    plt.clf()

    for i in range(runs):
        for j in range(confs):
            plt.scatter(np.arange(1, cases + 1), normalizedJoinedTimes[i,:,j], label="{} - Configuration {}".format(labels[i], j + 1))

    plt.legend()
    plt.xticks(np.arange(1, cases + 1))
    plt.xlim(0, cases + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Experiment Times Scatter Plot')
    plt.xlabel('Experiment (#)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Times Scatter Plot')))
    plt.clf()

    for i in range(runs):
        for j in range(confs):
            plt.scatter(np.arange(1, cases + 1), normalizedAccuracies[i,:,j], label="{} - Configuration {}".format(labels[i], j + 1))

    plt.legend()
    plt.xticks(np.arange(1, cases + 1))
    plt.xlim(0, cases + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Experiment Accuracy Scatter Plot')
    plt.xlabel('Experiment (#)')
    plt.ylabel('Normalized Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Accuracy Scatter Plot')))
    plt.clf()

    for i in range(runs):
        for j in range(confs):
            plt.scatter(np.arange(1, cases + 1), normalizedOverallAccuracies[i,:,j], label="{} - Configuration {}".format(labels[i], j + 1))

    plt.legend()
    plt.xticks(np.arange(1, cases + 1))
    plt.xlim(0, cases + 1)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Experiment Overall Accuracy Scatter Plot')
    plt.xlabel('Experiment (#)')
    plt.ylabel('Normalized Overall Accuracy (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Experiment Overall Accuracy Scatter Plot')))
    plt.clf()

    for i in range(runs):
        for j in range(confs):
            plt.scatter(normalizedAccuracies[i,:,j], normalizedJoinedTimes[i,:,j], label="{} - Configuration {}".format(labels[i], j + 1))

    plt.legend()
    plt.xticks(np.arange(0, 101, 10))
    plt.xlim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Accuracy Times Scatter Plot')
    plt.xlabel('Normalized Accuracy (%)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Accuracy Times Scatter Plot')))
    plt.clf()

    for i in range(runs):
        for j in range(confs):
            plt.scatter(normalizedOverallAccuracies[i,:,j], normalizedJoinedTimes[i,:,j], label="{} - Configuration {}".format(labels[i], j + 1))

    plt.legend()
    plt.xticks(np.arange(0, 101, 10))
    plt.xlim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.title('Overall Accuracy Times Scatter Plot')
    plt.xlabel('Normalized Overall Accuracy (%)')
    plt.ylabel('Normalized Joined Times (%)')
    plt.savefig(os.path.join(os.path.join(outputDir, 'Overall Accuracy Times Scatter Plot')))
    plt.clf()
                
def main(args):
    results = args[:2]
    labels = args[2:4]
    conf = int(args[4])
    accuracy = int(args[5])
    overallAccuracy = int(args[6])
    jobTime = int(args[7])
    taskTime = int(args[8])
    outputDir = args[-1]

    accuracies, overallAccuracies, jobTimes, taskTimes = getData(results, conf, accuracy, overallAccuracy, jobTime, taskTime)
    print("Generating comparison plots...")
    scatterPlots(accuracies, overallAccuracies, jobTimes, taskTimes, labels, outputDir)
    print("Done.")

if __name__ == '__main__':    
    main(sys.argv[1:])