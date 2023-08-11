import cv2

bikes_classifier = cv2.CascadeClassifier('Cascades/Vehicle and pedestrain detection/cars.xml')

camera = cv2.VideoCapture('cars.mp4')

while (True):

    ret, img = camera.read()

    blur = cv2.blur(img, (3, 3))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    bikes = bikes_classifier.detectMultiScale(gray)

    for (x, y, w, h) in bikes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()
camera.release()