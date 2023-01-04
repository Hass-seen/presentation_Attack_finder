import csv
import os


print(vid.rsplit(".",1)[0]+".pt")

data=[vid.rsplit(".",1)[0]+".pt",2]

with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for row in data:
        writer.writerow(row)