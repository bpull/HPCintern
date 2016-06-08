import csv
from subprocess import Popen, PIPE, STDOUT

weight1 = [50, 100, 200, 300, 400, 500]
weight2 = [50, 100, 200, 300, 400, 500]
step = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

best_w1 = 50
best_w2 = 50
best_st = 1e-1
max_lrn = 0

f = open("out.csv", 'a')

for w1 in weight1:
    for w2 in weight2:
        for s in step:
            p=Popen(['python','tensorbaby.py',str(w1),str(w2),str(s)], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
            output = p.communicate()[0]
            f.write(str(output)+","+str(w1)+","+str(w2)+","+repr(s)+"\n")

with open('out.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        cur_lrn = row[0]
        cur_lrn = float(cur_lrn)
        if cur_lrn > max_lrn:
            best_w1 = row[1]
            best_w2 = row[2]
            best_st = row[3]

print("Best network of 2 layers have %s neurons first and %s neurons second with learning rate %s"%(best_w1, best_w2, best_st))
