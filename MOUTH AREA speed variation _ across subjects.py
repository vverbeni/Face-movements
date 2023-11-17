#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy import interpolate
import json
import cv2
import os
import scipy.io.wavfile as wav

def resample_data(data, time_points, frequency, mode="cubic"):
    """Resamples non-uniform data to a uniform time series according to a specific frequency."""
    np_data = np.array(data)
    np_time_points = np.array(time_points)
    interp = interpolate.interp1d(np_time_points, np_data, kind=mode)

    resampled_time_points = np.arange(min(time_points), max(time_points), 1 / frequency)
    resampled_data = interp(resampled_time_points)
    return resampled_data, resampled_time_points

root_dir = r'C:... GESTURE'
folders = os.listdir(root_dir)

participant_dict= {}

for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue  # Skip non-directory items
    
    print("Participant:", folder)
    
    
    subfolders = os.listdir(folder_path)
    
    R_dict = {}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  
        
        print ("File:", subfolder)
        
        
        file_list = os.listdir(subfolder_path)
        num_files = len(file_list)

        timestamps = []
        mouth_landmarks = [[] for _ in range(8)]  # Create an empty list for each mouth landmark across frames (mouth indices: 0-7).

        # Creating artificial timestamp 0
        initial_timestamp = 0
        timestamps.append(initial_timestamp)
        
        # Creating a copy of each landmark's coordinates at frame 0, to put in first position within mouth_landmarks's sublists
        json_file = os.path.join(subfolder_path, 'frame_{}.json'.format(0))
        with open(file=json_file, mode="r", encoding="utf-16-le") as read_file:
            data = json.load(read_file)
        mouth_points = [tuple(data["Bodies"][0]["Face"]["Points2D"][i].values()) for i in range(60, 68)]
        for k, point in enumerate(mouth_points):
            mouth_landmarks[k].append(point)

        # Looping through all frames in a given subfolder
        for i in range(num_files):
            # Load the json file for a frame
            json_file = os.path.join(subfolder_path, 'frame_{}.json'.format(i))
            with open(file=json_file, mode="r", encoding="utf-16-le") as read_file:
                data = json.load(read_file)

            # Getting the timestamp of the current frame
            timestamp = data["Timestamp"]  
            timestamps.append(timestamp)

            # Get the MOUTH landmarks from the json data (and converting from dict. to tuples)
            mouth_points = [tuple(data["Bodies"][0]["Face"]["Points2D"][i].values()) for i in range(60, 68)]
            for k, point in enumerate(mouth_points):
                mouth_landmarks[k].append(point)
                
        # Creating artificial final timestamp, taken from the associated wav file (in case it's a few milliseconds longer than Kinect's data)
        audios_path = r'C:/... AUDIO'
        subject_folder = os.path.join(audios_path, folder)
        
        wav_files = [f for f in os.listdir(subject_folder) if f.endswith('.wav')]
        if subfolder + '.wav' in wav_files:
            audio_file = os.path.join(subject_folder, subfolder + '.wav')
            sample_rate, audio_data = wav.read(audio_file)
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]
            duration = len(audio_data) / sample_rate
            timestamps.append(duration)

        # Creating a copy of each landmark's coords in the last frame , to put in the final position within mouth_landmarks' sublists
        last_frame_json_file = os.path.join(subfolder_path, 'frame_{}.json'.format(num_files - 1))
        with open(last_frame_json_file, mode="r", encoding="utf-16-le") as read_file:
            data = json.load(read_file)
        mouth_points = [tuple(data["Bodies"][0]["Face"]["Points2D"][i].values()) for i in range(60, 68)]
        for k, point in enumerate(mouth_points):
            mouth_landmarks[k].append(point)
            
        # Resampling each landmark list
        resampled_timestamps = []
        resampled_landmarks = []
        for landmarks in mouth_landmarks:
            landmark_x = [point[0] for point in landmarks]
            landmark_y = [point[1] for point in landmarks]

            resampled_x_data, resampled_time_points = resample_data(data=landmark_x, time_points=timestamps, frequency=8, mode="cubic")
            resampled_y_data, _ = resample_data(data=landmark_y, time_points=timestamps, frequency=8, mode="cubic")

            resampled_landmarks.append(resampled_x_data)
            resampled_landmarks.append(resampled_y_data)
            resampled_timestamps.append(resampled_time_points)

        import pandas as pd

        # Combine the x and y coordinates for each landmark into a single DataFrame
        df = pd.DataFrame(resampled_landmarks)

        # Transpose the DataFrame to have landmarks as columns and frames as rows
        df = df.transpose()

        # Convert each row of the DataFrame into a list of coupled coordinates
        coupled_coordinates = df.apply(lambda row: list(row), axis=1).tolist()

        # Select one sublist from resampled_timestamps
        new_timestamps = resampled_timestamps[0]

        mouth_areas = []
        for landmark_coords in coupled_coordinates:
            # Extract x and y coordinates from each sublist in coupled_coordinates
            landmark_x = landmark_coords[0::2]
            landmark_y = landmark_coords[1::2]

            # Create a list of points for the mouth contour
            mouth_contour_points = [(x, y) for x, y in zip(landmark_x, landmark_y)]

            # Convert the contour points to a numpy array
            mouth_contour = np.array(mouth_contour_points, dtype=np.int32)

            # Calculate the area of the mouth contour
            mouth_area = cv2.contourArea(mouth_contour)
            mouth_areas.append(mouth_area)

        mouth_area_diffs = [abs(mouth_areas[n] - mouth_areas[n-1]) for n in range(1, len(mouth_areas))]

        timestamp_diffs = [abs(new_timestamps[m] - new_timestamps[m-1]) for m in range(1, len(new_timestamps))]

        speeds = []
        for area_diff, time_diff in zip(mouth_area_diffs, timestamp_diffs):
            if time_diff != 0:  # to avoid division error by zero if a time_diff is 0
                speed = area_diff / time_diff
            else:
                speed = 0
            speeds.append(speed)
        
        # Create the dictionary with sequential keys for speed coordinates
        speed_dict = {f"speed_{i+1}": coordinate for i, coordinate in enumerate(speeds)}
        
        # Get the subfolder name (without the extension)
        R_subfolder_name = os.path.splitext(subfolder)[0]

        # Add the speed dictionary to the subfolder dictionary
        R_dict[R_subfolder_name] = speed_dict

    # Add the subfolder dictionary to the main folder dictionary
    participant_dict[folder] = R_dict

# Save the folder hierarchy as a JSON file
json_file_path = 'Speed_samples_improved_MOUTH.json'  # Specify the file path for the JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(participant_dict, json_file, indent=4)

print("JSON file created:", json_file_path)

