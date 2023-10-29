import tensorflow as tf
import streamlit as st
import numpy as np
import tempfile
import cv2

class deploy:
    def __init__(self):
        pass

    def ImageBox(self, image):
        new_shape=(640, 640)
        color=(255, 0, 0)
        
        width, height, channel = image.shape
        
        ratio = min(new_shape[0] / width, new_shape[1] / height)
        
        new_unpad = int(round(height * ratio)), int(round(width * ratio))
        dw, dh = (new_shape[0] - new_unpad[0])/2, (new_shape[1] - new_unpad[1])/2

        if (height, width) != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return image, ratio, (dw, dh)


    def BoxScore(self, boxes, dwdh, ratio):
        new_boxes = []
        
        for box in boxes:
            x0,y0,x1,y1 = box
            box = np.array([x0-dwdh[0], y0-dwdh[1], x1-dwdh[0], y1-dwdh[1]])/ratio
            box = box.round().astype(np.int32)
                    
            new_boxes.append(box)

        return np.array(new_boxes)

    
    def BoxScore2(self, output_data):
        boxes, scores = [],[]

        for i, (x0,y0,x1,y1, oscore) in enumerate(output_data):

            for j in range(len(x0)):
                if oscore[j] >= 0.8:
                    box = np.array([x0[j], y0[j], x1[j], y1[j]])
                    box = box.round().astype(np.int32).tolist()
                    
                    score = round(float(oscore[j]), 3)
                    
                    if box not in boxes:
                        boxes.append(box)
                        scores.append(score)

        return np.array(boxes),np.array(scores)

    
    def image_preprocessing(self, image):
        img = np.expand_dims(image, axis=0)/255.0
        img = np.transpose(img, (0,3,1,2))
        img = np.ascontiguousarray(img)
        input_data = img.astype(np.float32)
        return input_data


    def compute_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        x_intersection = max(0, min(x1_min, x2_min) - max(x1_max, x2_max))
        y_intersection = max(0, min(y1_min, y2_min) - max(y1_max, y2_max))
        intersection_area = x_intersection * y_intersection

        area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
        area_box2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        iou = intersection_area / (area_box1 + area_box2 - intersection_area)
        return iou


    def nms(self, boxes, scores, iou_threshold):
        selected_indices = []

        sorted_indices = np.argsort(scores)[::-1]

        while len(sorted_indices) > 0:
            best_index = sorted_indices[0]
            selected_indices.append(best_index)

            iou_values = [self.compute_iou(boxes[best_index], boxes[idx]) for idx in sorted_indices[1:]]
            filtered_indices = np.where(np.array(iou_values) < iou_threshold)[0]
            sorted_indices = sorted_indices[filtered_indices + 1]

        return selected_indices


    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y                                                                                             

    
    def run(self, video_path, input_details, output_data):

        camera = cv2.VideoCapture(video_path)
        FRAME_WINDOW = st.image([])
      
        while True:
            
            ret, frame = camera.read()
            print(frame.shape)
            if not ret or camera.isOpened()==False:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preprocessed_image, ratio, (dw,dh) = self.ImageBox(frame)
            input_data = self.image_preprocessing(preprocessed_image)

            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]["index"])
            

            boxes, scores = self.BoxScore2(output_data)
            indices = self.nms(boxes, scores, 0.3)
	
            if len(indices)==0:
                continue

            boxes = self.xywh2xyxy(boxes[indices])
           
            for (bbox, score) in zip(self.BoxScore(boxes,(dw,dh),ratio), scores[indices]):
                bbox = bbox.round().astype(np.int32).tolist()
                cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), (0,255,0), 1)
            FRAME_WINDOW.image(frame)

        camera.release()


if __name__=="__main__":
    
    model_path = "model.tflite"
    user = "admin"
    password = "Wstation0707!"
    ip_address = "192.168.100.99"
    rtsp = 554

    camera = f"rtsp://{user}:{password}@{ip_address}:{rtsp}/h264/ch1/main/av_stream"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    deployment = deploy()
    deployment.run(camera,input_details,output_details)


