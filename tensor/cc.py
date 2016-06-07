#!/usr/bin/python
import csv
import h5py
import numpy as np
from random import randrange

def doit(f,a,i):
    f.write(str(a[i]))
    f.write(",")

def doitlast(f,a,i):
    f.write(str(a[i]))
    f.write('\n')


count = 0
tcount= 0
limit_bal = []
sex       = []
education1= []
education2= []
education3= []
education4= []
marriage1 = []
marriage2 = []
marriage3 = []
age       = []
pay_1     = []
pay_2     = []
pay_3     = []
pay_4     = []
pay_5     = []
pay_6     = []
bill_amt1 = []
bill_amt2 = []
bill_amt3 = []
bill_amt4 = []
bill_amt5 = []
bill_amt6 = []
pay_amt1  = []
pay_amt2  = []
pay_amt3  = []
pay_amt4  = []
pay_amt5  = []
pay_amt6  = []
default   = []

tlimit_bal = []
tsex       = []
teducation1= []
teducation2= []
teducation3= []
teducation4= []
tmarriage1 = []
tmarriage2 = []
tmarriage3 = []
tage       = []
tpay_1     = []
tpay_2     = []
tpay_3     = []
tpay_4     = []
tpay_5     = []
tpay_6     = []
tbill_amt1 = []
tbill_amt2 = []
tbill_amt3 = []
tbill_amt4 = []
tbill_amt5 = []
tbill_amt6 = []
tpay_amt1  = []
tpay_amt2  = []
tpay_amt3  = []
tpay_amt4  = []
tpay_amt5  = []
tpay_amt6  = []
tdefault   = []

with open('credit.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:

        randint = randrange(3)

        if count < 2:
            count+=1
        else:
            #2nd column is amount of given credit
            if randint < 2:
                count+=1

                limit_bal.append(row[1])
                sex.append(int(row[2])-1)

                if row[3] == "1":
                    education1.append("1")
                    education2.append("0")
                    education3.append("0")
                    education4.append("0")
                elif row[3] == "2":
                    education1.append("0")
                    education2.append("1")
                    education3.append("0")
                    education4.append("0")
                elif row[3] == "3":
                    education1.append("0")
                    education2.append("0")
                    education3.append("1")
                    education4.append("0")
                else:
                    education1.append("0")
                    education2.append("0")
                    education3.append("0")
                    education4.append("1")

                if row[4] == "1":
                    marriage1.append("1")
                    marriage2.append("0")
                    marriage3.append("0")
                elif row[4] == "2":
                    marriage1.append("0")
                    marriage2.append("1")
                    marriage3.append("0")
                else:
                    marriage1.append("0")
                    marriage2.append("0")
                    marriage3.append("1")

                age.append(row[5])
                pay_1.append(row[6])
                pay_2.append(row[7])
                pay_3.append(row[8])
                pay_4.append(row[9])
                pay_5.append(row[10])
                pay_6.append(row[11])
                bill_amt1.append(row[12])
                bill_amt2.append(row[13])
                bill_amt3.append(row[14])
                bill_amt4.append(row[15])
                bill_amt5.append(row[16])
                bill_amt6.append(row[17])
                pay_amt1.append(row[18])
                pay_amt2.append(row[19])
                pay_amt3.append(row[20])
                pay_amt4.append(row[21])
                pay_amt5.append(row[22])
                pay_amt6.append(row[23])
                default.append(row[24])

            else:
                tlimit_bal.append(row[1])
                tsex.append(row[2])

                if row[3] == "1":
                    teducation1.append("1")
                    teducation2.append("0")
                    teducation3.append("0")
                    teducation4.append("0")
                elif row[3] == "2":
                    teducation1.append("0")
                    teducation2.append("1")
                    teducation3.append("0")
                    teducation4.append("0")
                elif row[3] == "3":
                    teducation1.append("0")
                    teducation2.append("0")
                    teducation3.append("1")
                    teducation4.append("0")
                else:
                    teducation1.append("0")
                    teducation2.append("0")
                    teducation3.append("0")
                    teducation4.append("1")

                if row[4] == "1":
                    tmarriage1.append("1")
                    tmarriage2.append("0")
                    tmarriage3.append("0")
                elif row[4] == "2":
                    tmarriage1.append("0")
                    tmarriage2.append("1")
                    tmarriage3.append("0")
                else:
                    tmarriage1.append("0")
                    tmarriage2.append("0")
                    tmarriage3.append("1")


                tage.append(row[5])
                tpay_1.append(row[6])
                tpay_2.append(row[7])
                tpay_3.append(row[8])
                tpay_4.append(row[9])
                tpay_5.append(row[10])
                tpay_6.append(row[11])
                tbill_amt1.append(row[12])
                tbill_amt2.append(row[13])
                tbill_amt3.append(row[14])
                tbill_amt4.append(row[15])
                tbill_amt5.append(row[16])
                tbill_amt6.append(row[17])
                tpay_amt1.append(row[18])
                tpay_amt2.append(row[19])
                tpay_amt3.append(row[20])
                tpay_amt4.append(row[21])
                tpay_amt5.append(row[22])
                tpay_amt6.append(row[23])
                tdefault.append(row[24])


limit_bal = np.array(limit_bal,dtype=np.int)
sex       = np.array(sex,dtype=np.int)
education1= np.array(education1,dtype=np.int)
education2= np.array(education2,dtype=np.int)
education3= np.array(education3,dtype=np.int)
education4= np.array(education4,dtype=np.int)
marriage1 = np.array(marriage1,dtype=np.int)
marriage2 = np.array(marriage2,dtype=np.int)
marriage3 = np.array(marriage3,dtype=np.int)
age       = np.array(age,dtype=np.int)
pay_1     = np.array(pay_1,dtype=np.int)
pay_2     = np.array(pay_2,dtype=np.int)
pay_3     = np.array(pay_3,dtype=np.int)
pay_4     = np.array(pay_4,dtype=np.int)
pay_5     = np.array(pay_5,dtype=np.int)
pay_6     = np.array(pay_6,dtype=np.int)
bill_amt1 = np.array(bill_amt1,dtype=np.int)
bill_amt2 = np.array(bill_amt2,dtype=np.int)
bill_amt3 = np.array(bill_amt3,dtype=np.int)
bill_amt4 = np.array(bill_amt4,dtype=np.int)
bill_amt5 = np.array(bill_amt5,dtype=np.int)
bill_amt6 = np.array(bill_amt6,dtype=np.int)
pay_amt1  = np.array(pay_amt1,dtype=np.int)
pay_amt2  = np.array(pay_amt2,dtype=np.int)
pay_amt3  = np.array(pay_amt3,dtype=np.int)
pay_amt4  = np.array(pay_amt4,dtype=np.int)
pay_amt5  = np.array(pay_amt5,dtype=np.int)
pay_amt6  = np.array(pay_amt6,dtype=np.int)
default   = np.array(default,dtype=np.int)

tlimit_bal = np.array(tlimit_bal,dtype=np.int)
tsex       = np.array(tsex,dtype=np.int)
teducation1= np.array(teducation1,dtype=np.int)
teducation2= np.array(teducation2,dtype=np.int)
teducation3= np.array(teducation3,dtype=np.int)
teducation4= np.array(teducation4,dtype=np.int)
tmarriage1 = np.array(tmarriage1,dtype=np.int)
tmarriage2 = np.array(tmarriage2,dtype=np.int)
tmarriage3 = np.array(tmarriage3,dtype=np.int)
tage       = np.array(tage,dtype=np.int)
tpay_1     = np.array(tpay_1,dtype=np.int)
tpay_2     = np.array(tpay_2,dtype=np.int)
tpay_3     = np.array(tpay_3,dtype=np.int)
tpay_4     = np.array(tpay_4,dtype=np.int)
tpay_5     = np.array(tpay_5,dtype=np.int)
tpay_6     = np.array(tpay_6,dtype=np.int)
tbill_amt1 = np.array(tbill_amt1,dtype=np.int)
tbill_amt2 = np.array(tbill_amt2,dtype=np.int)
tbill_amt3 = np.array(tbill_amt3,dtype=np.int)
tbill_amt4 = np.array(tbill_amt4,dtype=np.int)
tbill_amt5 = np.array(tbill_amt5,dtype=np.int)
tbill_amt6 = np.array(tbill_amt6,dtype=np.int)
tpay_amt1  = np.array(tpay_amt1,dtype=np.int)
tpay_amt2  = np.array(tpay_amt2,dtype=np.int)
tpay_amt3  = np.array(tpay_amt3,dtype=np.int)
tpay_amt4  = np.array(tpay_amt4,dtype=np.int)
tpay_amt5  = np.array(tpay_amt5,dtype=np.int)
tpay_amt6  = np.array(tpay_amt6,dtype=np.int)
tdefault   = np.array(tdefault,dtype=np.int)


with open('learn.csv','wb') as f:
    for i in range(len(marriage1)):

        doit(f, limit_bal, i)
        doit(f, sex, i)
        doit(f, education1, i)
        doit(f, education2, i)
        doit(f, education3, i)
        doit(f, education4, i)
        doit(f, marriage1, i)
        doit(f, marriage2, i)
        doit(f, marriage3, i)
        doit(f, age, i)
        doit(f, pay_1, i)
        doit(f, pay_2, i)
        doit(f, pay_3, i)
        doit(f, pay_4, i)
        doit(f, pay_5, i)
        doit(f, pay_6, i)
        doit(f, bill_amt1, i)
        doit(f, bill_amt2, i)
        doit(f, bill_amt3, i)
        doit(f, bill_amt4, i)
        doit(f, bill_amt5, i)
        doit(f, bill_amt6, i)
        doit(f, pay_amt1, i)
        doit(f, pay_amt2, i)
        doit(f, pay_amt3, i)
        doit(f, pay_amt4, i)
        doit(f, pay_amt5, i)
        doit(f, pay_amt6, i)
        doitlast(f, default, i)

with open('test.csv', 'wb') as tf:
    for i in range(len(tmarriage1)):

        doit(tf,tlimit_bal, i)
        doit(tf,tsex, i)
        doit(tf,teducation1, i)
        doit(tf,teducation2, i)
        doit(tf,teducation3, i)
        doit(tf,teducation4, i)
        doit(tf,tmarriage1, i)
        doit(tf,tmarriage2, i)
        doit(tf,tmarriage3, i)
        doit(tf,tage, i)
        doit(tf,tpay_1, i)
        doit(tf,tpay_2, i)
        doit(tf,tpay_3, i)
        doit(tf,tpay_4, i)
        doit(tf,tpay_5, i)
        doit(tf,tpay_6, i)
        doit(tf,tbill_amt1, i)
        doit(tf,tbill_amt2, i)
        doit(tf,tbill_amt3, i)
        doit(tf,tbill_amt4, i)
        doit(tf,tbill_amt5, i)
        doit(tf,tbill_amt6, i)
        doit(tf,tpay_amt1, i)
        doit(tf,tpay_amt2, i)
        doit(tf,tpay_amt3, i)
        doit(tf,tpay_amt4, i)
        doit(tf,tpay_amt5, i)
        doit(tf,tpay_amt6, i)
        doitlast(tf,tdefault, i)
