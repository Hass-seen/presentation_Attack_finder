import torch 
import cv2
import os

folder_path = r"D:\db\training\real"

# Get a list of all the files in the folder
file_list = os.listdir(folder_path)
print(file_list)
for vid in file_list:
    video = cv2.VideoCapture("D:/db/training/real/"+vid)

    # Initialize a list to store the frames
    frames = []

    # Loop over the frames in the video
    for i in range(200):
        # Read the next frame
        success, frame = video.read()
        if not success:
            break
        
        # Preprocess the frame
        # frame = cv2.resize(frame, (224, 224))  # Resize the frame
        frame = frame.transpose((2, 0, 1))  # Transpose the frame to match the input dimensions of the network
        frame = torch.from_numpy(frame).float()  # Convert the frame to a tensor
        frame = frame.unsqueeze(0)  # Add a batch dimension to the frame
        print(frame.shape)
        # Add the frame to the list of frames
        frames.append(frame)
        
    
    # Concatenate the frames into a single tensor
    if len(frames) < 200:
        continue
    frames = torch.cat(frames, dim=0)

    print(frames.shape) #([100, 3, 240, 320])
    torch.save(frames,"D:/db/training_tesnsors/real_tensors/"+vid.rsplit(".",1)[0]+".pt")
