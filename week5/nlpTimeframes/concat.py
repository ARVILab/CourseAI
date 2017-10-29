import os

for i in range(1948, 2012):
    s = ''
    with open('../../datasets/nlp/subs/concat/%s.txt' % (i+2), 'w') as of:
        for j in range(i, i+5):
            fn = '../../datasets/nlp/subs/en/%s.txt' % j
            if os.path.isfile(fn):
                with open(fn) as f:
                    of.write(f.read())
        print(i)
