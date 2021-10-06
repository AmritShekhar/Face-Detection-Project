import cv2
import os
import numpy as np
import face_reco



path = 'assets/images'

identifiers = []
img = []

array = os.listdir(path)
print(array)

for ele in array:

    curImg = cv2.imread(f'{path}/{ele}')

    img.append(curImg)

    identifiers.append(os.path.splitext(ele)[0])

print(identifiers)


def encodeFaces(img):

    encodeArray = []

    for i in img:

        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)

        encode = face_reco.face_encodings(i)[0]

        encodeArray.append(encode)

    return encodeArray


print('processing...')
dataset = encodeFaces(img)

print('encoding completed')


seize = cv2.VideoCapture(0)

while True:
    signal, pic = seize.read()

    re_pic = cv2.resize(pic, (0, 0), None, 0.50, 0.50)

    re_pic = cv2.cvtColor(re_pic, cv2.COLOR_BGR2RGB)

    instSeize = face_reco.face_locations(re_pic)

    instEncode = face_reco.face_encodings(re_pic, instSeize)

    for encodeData, dataPos in zip(instEncode, instSeize):

        compare = face_reco.compare_faces(dataset, encodeData)

        disMeasure = face_reco.face_distance(dataset, encodeData)

        print(disMeasure)

        compElement = np.argmin(disMeasure)

        if compare[compElement]:
            value = identifiers[compElement].upper()
            print(value)

            y1, x2, y2, x1 = dataPos

            y1, x2, y2, x1 = 2*y1, 2*x2, 2*y2, 2*x1

            cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 255, 0), 1)

            cv2.rectangle(pic, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)

            cv2.putText(pic, value, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

    cv2.imshow('Webcam', pic)

    cv2.waitKey(1)
