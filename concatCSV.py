import os
import glob
import pandas as pd

if __name__ == '__main__':
    # open the file for writing
    f = open('data/output.csv', 'w')

    # loop through each file in data
    for file in os.listdir("data"):
        filename = os.fsdecode(file)
        print("\n\n" + filename + "\n\n")
        # open the file
        l = open("data/" + filename,"r")

        lines = l.readlines()

        # loop through each line
        for line in lines:
            digits = line.split(" ")

            # loop through each digit in a line
            for digit in digits:
                    # add that digit to the new file
                    f.write(digit)
                    f.write(" ")
                    print(digit, end='')


