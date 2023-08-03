import face_recognition
import cv2
import numpy as np
import csv 
import os
from datetime import datetime

video_capture=cv2.VideoCapture(0)

Tanmay_image = face_recognition.load_image_file('E:/project_face/image/known/Tanmay.jpg')
Tanmay_encoding = face_recognition.face_encodings(Tanmay_image)[0]

vardhan_image = face_recognition.load_image_file('E:/project_face/image/known/vardhan.jpg')
vardhan_encoding = face_recognition.face_encodings(vardhan_image)[0]

Abhishek_image = face_recognition.load_image_file('E:/project_face/image/known/Abhishek.jpg')
Abhishek_encoding = face_recognition.face_encodings(Abhishek_image)[0]

sonam_image = face_recognition.load_image_file('E:/project_face/image/known/sonam.jpg')
sonam_encoding = face_recognition.face_encodings(sonam_image)[0]

known_face_encoding=[
Tanmay_encoding,
vardhan_encoding,
Abhishek_encoding,
sonam_encoding
]

known_faces_names = [
    "Tanmay",
    "vardhan",
    "Abhishek",
    "sonam"
]

students = known_faces_names.copy()

face_locations = []
face_encodings =[]
face_names = []
s= True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d") 

f= open(current_date+'python_Attendance.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    _,frame= video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.75,fy=0.75)
    rgb_small_frame = small_frame[:, :, ::-1]
    lnwriter.writerow(['Name','Time','present/Absent'])

    if s: 
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = [] 
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance =face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name= known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    print("Attendence marked for: "+ name)
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time,'present'])

    cv2.imshow("attendence system", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break            

video_capture.release()
cv2.destroyAllWindows()
f.close()