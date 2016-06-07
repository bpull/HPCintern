import csv

with open('learn.csv', 'rb') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        i=i+1
    print i

with open('test.csv', 'rb') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        i=i+1
    print i
