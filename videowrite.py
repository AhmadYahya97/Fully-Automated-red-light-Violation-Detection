import cv2
writer = None
vs = cv2.VideoCapture('videos/aziz1.mp4')
fps = vs.get(cv2.CAP_PROP_FPS)
print(fps)
while True:
    (grabbed, frame) = vs.read()
    if(not grabbed):
        break
    else:
        frame = cv2.resize(frame, (1000, 750))
        cv2.imshow('fr',frame)
        cv2.waitKey(






        )
        if writer is None:
            # initialize our video writer

            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('hi.avi', fourcc, fps



                                     ,
                                     (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)

writer.release()






