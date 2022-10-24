# -*- coding: utf-8 -*-

import cv2

OPENCV_OBJECT_TRACKERS = {"csrt"      : cv2.TrackerCSRT_create,
		                  "kcf"       : cv2.TrackerKCF_create,
		                  "boosting"  : cv2.legacy.TrackerBoosting_create,
		                  "mil"       : cv2.TrackerMIL_create,
		                  "tld"       : cv2.legacy.TrackerTLD_create,
		                  "medianflow": cv2.legacy.TrackerMedianFlow_create,
		                  "mosse"     : cv2.legacy.TrackerMOSSE_create}

tracker_name = "boosting"

trackers = cv2.legacy.MultiTracker_create()
# çoklu nesne takibi algoritması

video_path = "MOT17-04-DPM.mp4"
# vidomuzun yolu
cap = cv2.VideoCapture(video_path)

fps = 30     
# videomuzun fpsini videoyu indirdiğimiz sitede
# buluyoruz.
f = 0
# f de frame sayısını tutuyoruz.
while True:
    
    ret, frame = cap.read()
    # capture ettiğimiz videoyu okuyoruz.
    (H, W) = frame.shape[:2]
    # yükseklik ve genişliğimi alırım.
    frame = cv2.resize(frame, dsize = (960, 540))
    
    (success , boxes) = trackers.update(frame)
    # tracker ımı update edicem.
    # update sonucunda success ve boxes larımı return
    # edicek.
    info = [("Tracker", tracker_name),
        	("Success", "Yes" if success else "No")]
    
    string_text = ""
    # bunun içerisine görselleştirmek istediğimiz 
    # şeyleri yazıcaz.
    
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        string_text = string_text + text + " "
    
    cv2.putText(frame, string_text, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # şimdi bunları yazdıralım. 10 a 20 lik bir yere 
    # yazdırsın.
    
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # kutularımızı görselleştirelim.
    # x, y, w ve h leri ilk başta int e çeviriyoruz.
    # sonra da frame imizin üzerine ekleyerek görselleştiriyoruz.
    
    cv2.imshow("Frame", frame)
    # sonra frame imizi görselleştiriyoruz.
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("t"):
        
        box = cv2.selectROI("Frame", frame, fromCenter=False)
    
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        trackers.add(tracker, frame, box)
        # seçilen kutucuklar tracker ın içerisine ekliyoruz ki
        # tracker ımız bunları takip etsin.
    elif key == ord("q"):break

    f = f + 1
    # frame imizi 1 arttırıyoruz.
    
cap.release()
cv2.destroyAllWindows() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
