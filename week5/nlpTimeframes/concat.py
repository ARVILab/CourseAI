import os

with open('../../datasets/nlp/subs/concat/all.txt', 'w') as allfile:
    for i in range(1948, 2002):
        s = ''
        #with open('../../datasets/nlp/subs/concat/%s.txt' % (i+2), 'w') as of:
        for j in range(i, i+5):
            fn = '../../datasets/nlp/subs/en/%s.txt' % j
            if os.path.isfile(fn):
                with open(fn,'r') as f:
                    l = f.read()
                #   of.write(l)
                    allfile.write(l)
        print(i)
