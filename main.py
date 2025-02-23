from cv2 import cv2
import face_recognition as fr

pic_a = fr.load_image_file('pictures/picture_a.jpg')

#web_cam_pic_a = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#[success, pic_a] = web_cam_pic_a.read()
'''
if success:
    face_a = fr.face_locations(image)[0]
else:
    print('Error!')
'''
pic_b = fr.load_image_file('pictures/picture_b.jpg')

# Turn pictures into RGB
pic_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2RGB)
pic_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2RGB)

#locate face
face_a = fr.face_locations(pic_a)[0]
face_b = fr.face_locations(pic_b)[0]

#Encode face
face_a_encoded = fr.face_encodings(pic_a)[0]
face_b_encoded = fr.face_encodings(pic_b)[0]

#Show Container
cv2.rectangle(pic_a, (face_a[3],face_a[0]), (face_a[1], face_a[2]), (0, 255, 0), 2)
cv2.rectangle(pic_b, (face_b[3],face_b[0]), (face_b[1], face_b[2]), (0, 255, 0), 2)

result = fr.compare_faces([face_a_encoded], face_b_encoded)
print(result)

#check how similar 2 faces are (distance)
distance = fr.face_distance([face_a_encoded], face_b_encoded)
print(distance)

cv2.putText(pic_a, f'Result: {result} {distance.round(2)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
cv2.imshow('Picture A', pic_a)
cv2.imshow('Picture B', pic_b)

cv2.waitKey(0)
