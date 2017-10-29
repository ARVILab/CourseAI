__author__ = 'oles'
import json
import numpy as np
years = []
startYear = 1950
lastYear = 2005
name = 'subtitlesEn'
for i in range(startYear, lastYear+1):
    with open('UI/static/3d/'+str(i)+'.obj', 'r') as yearfile:
        years.append([])
        for s in yearfile:
            if s[0] == 'v':
                v = s.split()[-3:]
                for k in v:
                    years[i-startYear].append(np.round(float(k), 4))

jsonObject = { 'name':name,
               'startYear':startYear,
                'lastYear':lastYear,
               'years':[]
}
for i in range(0,len(years)):
    jsonObject["years"].append(years[i])

json.dump(jsonObject, open('UI/models/'+name +'.json','w'))