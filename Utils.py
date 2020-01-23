"""
Helper functions for the project on scaphoid fractures
"""

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD
from pathlib import Path
import os
import shutil
import cv2

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/Code/Datasets/Scaphoid/'

rawCR_dir = home_dir + 'Raw_Scaphoid_CR/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()


def process_raw():

    """
    This function takes the 9883 raw DICOMs and removes non image files, then saves the DICOMs
    as filename: Accno_BodyPart/Side_View
    """

    # Load the filenames and randomly shuffle them
    save_path = home_dir + 'Cleaned_CR/'
    filenames = sdl.retreive_filelist('**', True, rawCR_dir)
    shuffle(filenames)

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0

    for file in filenames:

        # Load the Dicom
        try:
            header = sdl.load_DICOM_Header(file, multiple=False)
        except Exception as e:
            #print('DICOM Error: %s' %e)
            continue

        # Retreive the view information
        try:
            view = header['tags'].ViewPosition.upper()
            laterality = header['tags'].ImageLaterality.upper()
            part = header['tags'].BodyPartExamined.upper()
            accno = header['tags'].AccessionNumber
        except Exception as e:
            #print('Header error: %s' %e)
            continue

        """
            Some odd values that appear:
            view: TAN, LATERAL, NAVICULAR, LLO
            part: PORT WRIST, 
        """

        # Sometimes lateraity is off
        if laterality != 'L' and laterality != 'R':
            if 'RIGHT' in header['tags'].StudyDescription.upper(): laterality = 'R'
            elif 'LEFT' in header['tags'].StudyDescription.upper(): laterality = 'L'

        # Set destination filename info
        dst_File = accno + '_' + laterality + '-' + part + '_' + view

        # Skip non wrists or hands
        if 'WRIST' not in part and 'HAND' not in part:
            print ('Skipping: ', dst_File)
            continue

        # Filename
        savedir = (save_path + accno + '/')
        savefile = savedir + dst_File + '.dcm'
        if not os.path.exists(savedir): os.makedirs(savedir)
        #savefile = save_path + dst_File + '.dcm'

        # Copy to the destination folder
        shutil.copyfile(file, savefile)
        #if index % 10 == 0 and index > 1:
        print('Saving pt %s of %s to dest: %s' % (index, len(filenames), dst_File))

        # Increment counters
        index += 1

    print('Done with %s images saved' % index)


def raw_to_PNG():

    """
    This function takes the 9883 raw DICOMs and removes non image files, then saves the DICOMs as PNGs
    of filename: Accno_BodyPart/Side_View. The PNG format is used because it supports 16 bit grayscale
    """

    # Load the filenames and randomly shuffle them
    save_path = home_dir + 'PNG_Files/'
    filenames = sdl.retreive_filelist('**', True, rawCR_dir)
    shuffle(filenames)

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0

    for file in filenames:

        # Load the Dicom
        try:
            image, _, _, photometric, header = sdl.load_DICOM_2D(file)
            if photometric == 1: image *= -1
        except Exception as e:
            #print('DICOM Error: %s' %e)
            continue

        # Retreive the view information
        try:
            view = header['tags'].ViewPosition.upper()
            laterality = header['tags'].ImageLaterality.upper()
            part = header['tags'].BodyPartExamined.upper()
            accno = header['tags'].AccessionNumber
        except Exception as e:
            #print('Header error: %s' %e)
            continue

        image = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))

        """
            Some odd values that appear:
            view: TAN, LATERAL, NAVICULAR, LLO
            part: PORT WRIST, 
        """

        # Sometimes lateraity is off
        if laterality != 'L' and laterality != 'R':
            if 'RIGHT' in header['tags'].StudyDescription.upper(): laterality = 'R'
            elif 'LEFT' in header['tags'].StudyDescription.upper(): laterality = 'L'

        # Set destination filename info
        dst_File = accno + '_' + laterality + '-' + part + '_' + view

        # Skip non wrists or hands
        if 'WRIST' not in part and 'HAND' not in part:
            print ('Skipping: ', dst_File)
            continue

        # Filename
        # savedir = (save_path + accno + '/')
        # savefile = savedir + dst_File + '.png'
        # if not os.path.exists(savedir): os.makedirs(savedir)
        savefile = save_path + dst_File + '.png'

        # Save the image
        sdl.save_image(image, savefile)
        print('Saving pt %s of %s to dest: %s' % (index, len(filenames), dst_File))

        # Increment counters
        index += 1

    print('Done with %s images saved' % index)


#process_raw()
#raw_to_PNG()