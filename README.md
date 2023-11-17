# Face-movements
Easy-to-implement codes that facilitate the analysis of face kinematics data recorded via KINECT V2.


ALL CODES ASSUME THAT DATA ARE STORED IN A DIRECTORY IN WHICH FOLDERS ARE NESTED AS FOLLOWS -->

    KINEMATIC DATA FOLDER: { Participants sub-folders { Recordings sub-folders { frames (jsn files)

    AUDIO FOLDER: { Participants sub-folders { Recordings (audio files)


ALL PIECES OF CODE BUILD ON THE SYNCHRONIZATION BETWEEN AUDIO AND KINEMATIC DATA. IT IS IMPORTANT THAT SUB-FOLDERS/FILES BE NAMED CONSISTENLY ACROSS DATA TYPES:

    e.g. If a recording sub-folder within the KINEMATIC DATA FOLDER is named R01, the associated audio file within the AUDIO FOLDER must be named R01, too.
