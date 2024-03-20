import json


with open('face_acne/train/_annotations.createml.json') as f:
    d = json.load(f)
    print(d[0]['annotations'])

