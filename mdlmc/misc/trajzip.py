# coding=utf-8

import sys


def read_frame(file):
    atomnr = int(file.readline())
    file.readline()
    lines = []
    for i in range(atomnr):
        lines.append(file.readline()[:-1])
    return atomnr, "\n".join(lines)


def main(*args):
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    file1 = open(filename1, "r")
    file2 = open(filename2, "r")

    while True:
        try:
            n1, traj1 = read_frame(file1)
            n2, traj2 = read_frame(file2)
        except:
            break
        else:
            print(n1 + n2)
            print()
            print(traj1)
            print(traj2)


if __name__ == "__main__":
    main()
