
import cv2
import os
import numpy as np

class Dataset:
    def __init__(self):
        self.none = None
        self.Im_size = (64,64)
        self.sequence_length = 20
        self.directory = 'UCF50'
        self.class_list = self.get_classes()
        
        
    def extract_frames(self, path_to_video):
        frame_list = []

        cap = cv2.VideoCapture(path_to_video)
        video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(int(video_frames_count/self.sequence_length),1)
        for frame_counter in range(self.sequence_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * frame_interval)

            success, frame = cap.read()

            if not success:
                break

            normalized_frame = cv2.resize(frame,self.Im_size) / 255
            frame_list.append(normalized_frame)
        cap.release()
        return frame_list 
    
    def get_classes(self):
        return os.listdir(self.directory)

    def create(self):
        features = []
        labels = []
        paths_to_files = []

        for class_idx, class_name in enumerate(self.class_list):
            video_files_list = os.listdir(os.path.join(self.directory,class_name))

            for file_name in video_files_list:
                # get the frames from each video
                path_to_video = os.path.join(self.directory, class_name,file_name)
                frames = self.extract_frames(path_to_video)

                if len(frames) == self.sequence_length:
                    features.append(frames)
                    labels.append(class_idx) # labels in int
                    paths_to_files.append(path_to_video)

        return np.asarray(features), np.asarray(labels), paths_to_files
                    

# main 
        
data = Dataset()
dataset = data.create()

print(dataset[0])

