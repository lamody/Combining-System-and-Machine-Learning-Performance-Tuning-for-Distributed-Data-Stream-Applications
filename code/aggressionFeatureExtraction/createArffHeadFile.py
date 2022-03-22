import sys
import numpy as np

def main(arffs):
    attributes = []
    for i,arff in enumerate(arffs):
        print("Processing {}({}/{})".format(arff, i + 1, len(arffs)))
        with open(arff, "r") as f:
            data = False
            for line in f:
                if not data:
                    if "@data" in line:
                        data = True
                    elif (line and line != "" and "@attribute" in line):
                        tokens = line.strip().split()
                        t = "nominal" if tokens[2][0] == "{" and tokens[2][-1] == "}" else tokens[2]
                        if t == "nominal":
                            values = {}
                            for c in tokens[2][1:-1].split(","):
                                values[c] = 0
                        else:
                            values = []
                        attr = {
                                "name": tokens[1], 
                                "type": t,
                                "values": values
                            }
                        if i == 0:
                            attributes.append(attr) 
                else:
                    if (line.strip() != ""):
                        for i,a in enumerate(line.strip().split(",")):
                            if attributes[i]["type"] == "nominal":
                                attributes[i]["values"][a] += 1
                            else:
                                attributes[i]["values"].append(float(a))

    with open("result.head", "w") as f:
        f.write("% Features with min, max, average, stdev\n")
        for a in attributes:
            if a["type"] == "nominal":
                f.write("@attribute {} {} {}\n".format(a["name"], "{" + ",".join(a["values"].keys()) + "}", "{" + ",".join([str(x) for x in a["values"].values()]) + "}"))
            else:
                f.write("@attribute {} {} {} {} {} {}\n".format(a["name"], a["type"], np.min(a["values"]), np.max(a["values"]), np.mean(a["values"]), np.std(a["values"])))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: input arff file(s)")
        sys.exit(0)

    main(sys.argv[1:])