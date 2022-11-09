import numpy as np

count = 0

'''for powder1 in range(0, 18):
    count = count + 1
    for portion1 in range(1, 10, 1):
        for powder2 in range(1+powder1, 18):
            count = count + 1
            for portion2 in range(1, (10-portion1), 1):
                for powder3 in range(1+powder2, 18):
                    count = count + 1
'''

for powder1 in range(7, 18):
    count = count + 1
    for portion1 in range(1, 10, 1):
        for powder2 in range(1+powder1, 18):
            count = count + 1
            for portion2 in range(1, (10-portion1), 1):
                for powder3 in range(1+powder2, 18):
                    count = count + 1

print(count)