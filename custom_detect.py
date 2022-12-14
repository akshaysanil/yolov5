import torch
import cv2
import time
import re
import numpy as np
import easyocr


EASY_OCR = easyocr.Reader(['en'])
OCR_TH = 0.2

# fun to run detection
def detectx(frame, model):
    frame = [frame]
    print(f'[INFO] detecting')
    results = model(frame)


    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:,:-1]

    return labels, cordinates


# fun for plot bounding box and results
def plot_boxes(results, frame, classes):
    
    # This function takes results, frame and classes
    # results: contains labels and coordinates predicted by model on the given frame
    # classes: contains the strting labels

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f'[INFO] Total {n} detections...')
    print(f'[INFO] Looping through all detectins...')


    # looping through the detections
    for i in range(n):
        row = cord[i]

        # threshold value for detection 
        if row[4] >= 0.55 :
            print('[INFO] Extracting bounding box coordinates...')
            # bounding box cordinates
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) 
            text_d = classes[int(labels[i])]


            coords = [x1, y1, x2, y2]

            plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)


            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2) # Bbox
            cv2.rectangle(frame, (x1,y1-20), (x2,y1), (0, 255, 0 ), -1) # for text label background
            cv2.putText(frame, f'{plate_num}',(x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)


    return frame



# fun to recognize number plate numbers using easyocr
def recognize_plate_easyocr(img, coords, reader, region_threshold):
    #seperate cordinates from bounding box
    xmin, ymin, xmax, ymax = coords
    # coping the number plate fromthe whole image
    nplate = img[int(ymin) : int(ymax), int(xmin): int(xmax)] 


    ocr_result = reader.readtext(nplate)


    text = filter_text(region =nplate, ocr_result = ocr_result, region_threshold= region_threshold)

    if len(text) == 1:
        text = text[0].upper()
    return text


# to filter out wrong detections
def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]

    plate = []
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate



# main function

def main(img_path=None, vid_path=None, vid_out=None):
    
    print(f'[INFO] Loading model...')
    # loading the cutom trained model
    # the repo is stored locally
    model = torch.hub.load('./yolov5','custom',source ='local', path='best.pt',force_reload=True)

    classes  = model.names # class name in string format



    # for detection on image
    if img_path != None:
        print(f'[INFO] working with image : {img_path}')
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"


        frame = cv2.imread(img_path) # reding the image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results = detectx(frame, model=model) # detection happening here 

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = plot_boxes(results, frame, classes=classes)

        # creating a window to show the result
        cv2.namedWindow('img_only',cv2.WINDOW_NORMAL)

        while True:
            cv2.imshow('img_only',frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f'[INFO] exiting...')

                # to save the output result
                cv2.imwrite(f'{img_out_name}', frame)

                break


    elif vid_path != None:
        print(f'[INFO] working with video : {vid_path}')

        #reading the video 
        cap = cv2.VideoCapture(vid_path)

        # creating the video writer if the video capture path is given
        if vid_out:
            
            # by deafult videoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_out, codec, fps,(width, height))


        frame_no = 1

        cv2.namedWindow('vid_out',cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret and frame_no % 1 == 0:
                print(f'[INFO] working frame {frame_no}')

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = detectx(frame, model=model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame = plot_boxes(frame, results, classes=classes)

                cv2.imshow('vid_out ', frame)
                if vid_out:
                    print(f'[INFO] saving output video...')
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

                frame_no += 1

        print(f'[INFO] cleaing up ...')
        #releasing the writer
        out.release()

        #closing all windows 
        cv2.destroyAllWindows()


# calling the main function

main(img_path='./1.jpeg')