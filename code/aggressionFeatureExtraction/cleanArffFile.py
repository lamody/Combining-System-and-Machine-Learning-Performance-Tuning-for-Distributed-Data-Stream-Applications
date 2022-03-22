#################################################
#                                               #
#  This script takes one arff file as input     #
#  and cleans it by removing the the unwanted   #
#  comma (,) at the end of each line. The file  #
#  generated is <input_arff_file>.clean         #
#                                               #
#################################################

import sys

def main(arff):
    with open(arff, "r") as f:
        with open(arff + ".clean", "w") as t:
            for line in f:
                if (line == "" or line[0] in ("@", "%")):
                    t.write(line)
                else:
                    t.write(line.strip()[:-1] + "\n")

    print("Done")

if __name__ == "__main__":
    if len(sys.argv[1:]) != 1:
        print("Usage: input arff file")
        sys.exit(0)

    main(sys.argv[1:][0])
