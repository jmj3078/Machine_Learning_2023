import os

f = open('run.sh', 'w')
for i in range(0, 389):
    f.write(f"python ./poseLandmark_test.py -i ../for-student/test/{i}.jpg\n")