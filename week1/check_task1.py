true_labels = []
with open('../datasets/classifieds/labels.txt') as f:
    for s in f:
        s = s.strip().split(' ', 1)[1].split('_')[0]
        true_labels.append(int(s))

n = len(true_labels)
test_labels = [-1] * n

with open('test_my.txt') as f:
    for s in f:
        path, class_no = s.strip().split(' ')
        i = int(path.split('/')[-1].split('.')[0])
        test_labels[i] = int(class_no)

result = [true_labels[i] == test_labels[i] for i in range(n)]

c = sum(result)
print('accuracy: %.02f%%' % (100*c/n))
