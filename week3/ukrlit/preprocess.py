# -*- coding: utf-8 -*-

import os
import random
import re

paragraphs = []

to_space = ['\t', '†', '‡', '•', '™', '∞', ' ', '*', '$', '¦', '§', '®', '~']
to_empty = ['à', 'á', 'â', 'ç', 'è', 'é', 'ê', 'ó', 'ô', 'ö', 'ù', 'ü', 'ý', 'ą', 'ć', 'ę',
            'Ł', 'ł', 'ń', 'Ś', 'ś', 'ź', 'Ż', 'ż', '&amp;', '¤', '¬', '´', '&nbsp;', '&gt;', '&lt;', '&#246;', '&',
            'ά', 'ή', 'ί', 'ΰ', 'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'λ', 'μ', 'ν',
            'ξ', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ', 'υ', 'φ', 'χ', 'ό', 'ύ', 'ώ', 'ў', 'ѕ', 'ј', '√', '№',
            '°', '²', '½', '×', '̀', '́', '΄', '\xad']

for fn in os.listdir('texts'):
    t = open('texts/' + fn).read()
    for c in to_empty:
        t = t.replace(c, '')
    for c in to_space:
        t = t.replace(c, ' ')

    t = t.replace('\t', ' ')
    t = t.replace('…', '...')
    t = t.replace('„', '«')
    t = t.replace('“', '«')
    t = t.replace('”', '»')
    t = t.replace('’', "'")
    t = t.replace('‘', "'")
    t = t.replace('`', "'")
    t = t.replace('–', '-')

    t = t.replace('Ђ', 'е')
    t = t.replace('Α', 'А')
    t = t.replace('Ο', 'О')
    t = t.replace('Π', 'П')
    t = t.replace('Χ', 'Х')
    t = t.replace('ѣ', 'е')

    t = re.split(r'[\n|\r]', t)
    t = [s.strip() for s in t]
    paragraphs += [s for s in t if s]

random.shuffle(paragraphs)

text = []
for t in paragraphs:
    t = re.sub('-+', '-', t)
    t = re.sub('_+', ' ', t)
    t = re.sub('(\. )+', ' ', t)
    t = re.sub('\s+', ' ', t)
    text.append(t.strip())

print(len(text))

text = '\n'.join(text)

text = text.replace(' ,', ',').replace(' .', '.')

chars = list(set(text))
chars.sort()
print(len(chars))
print(chars)

# c = ''.join([c.encode('utf-8') for c in chars])
# with open('chars.txt', 'wb') as f:
#     f.write(c)

with open('text.txt', 'w') as f:
    f.write(text)
