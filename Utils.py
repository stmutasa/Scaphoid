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
    filenames = sdl.retreive_filelist('*', True, rawCR_dir)
    filenames2 = sdl.retreive_filelist('**', True, rawCR_dir)
    filenames += filenames2

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0
    saved = []

    for file in filenames:

        # Load the Dicom
        try:
            img, _, _, _, header = sdl.load_DICOM_2D(file)
        except Exception as e:
            #print('DICOM Error: %s' %e)
            continue

        # Retreive the view information. Only body part fail will fail us
        view, laterality, part, accno = 0, 0, 0, 0

        # Laterality: L, R or UK
        try:laterality = header['tags'].ImageLaterality.upper()
        except:
            try:laterality = header['tags'].Laterality.upper()
            except:
                try:
                    if 'LEFT' in header['tags'].StudyDescription.upper():laterality = 'L'
                    else: laterality = 'R'
                except:
                    try:
                        if 'LEFT' in header['tags'].SeriesDescription.upper():laterality = 'L'
                        else:laterality = 'R'
                    except: laterality = 'UKLAT'

        # Accession number
        try:
            dir = os.path.dirname(file)
            accno = dir.split('/')[-2]
        except:
            try: accno = header['tags'].StudyID
            except:
                try: accno = header['tags'].AccessionNumber
                except Exception as e:
                    print('Header error: %s' % e)
                    continue

        # View 'LAT, PA, OBL, TAN. Maybe use SeriesNumber 1 = PA, 2 = LAT if 2 views etc
        try: view = header['tags'].ViewPosition.upper()
        except:
            try: view = 'V' + str(header['tags'].SeriesNumber)
            except: view = 'UKVIEW'

        if not view:
            try: view = 'V' + header['tags'].SeriesNumber
            except: view = 'UKVIEW'

        # PART: WRIST, HAND
        try:
            if 'WRIST' in header['tags'].StudyDescription.upper():
                part = 'WRIST'
            elif 'HAND' in header['tags'].StudyDescription.upper():
                part = 'HAND'
            elif 'WRIST' in header['tags'].SeriesDescription.upper():
                part = 'WRIST'
            elif 'HAND' in header['tags'].SeriesDescription.upper():
                part = 'HAND'
            else:
                part = 'UKPART'
        except:
            try:
                part = header['tags'].BodyPartExamined.upper()
            except Exception as e:
                print('Header error: %s' % e)
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

        # Get instance number from number in this folder with these accnos
        savedir = (save_path + accno + '/')
        try: copy = len(sdl.retreive_filelist('dcm', True, savedir)) + 1
        except: copy = 1

        # Set destination filename info
        dst_File = accno + '_' + laterality + '-' + part + '_' + view + '-' + str(copy)

        # Skip non wrists or hands
        if 'WRIST' not in part and 'HAND' not in part:
            print ('Skipping: ', dst_File)
            continue

        # Filename
        savefile = savedir + dst_File + '.dcm'
        if not os.path.exists(savedir): os.makedirs(savedir)
        #savefile = save_path + dst_File + '.dcm'

        # Copy to the destination folder
        shutil.copyfile(file, savefile)
        #if index % 10 == 0 and index > 1:
        print('Saving pt %s of %s to dest: %s' % (index, len(filenames), dst_File))

        # Increment counters
        index += 1
        del img

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


def check_raw():

    """
    This function takes the 9883 raw DICOMs and removes non image files, then saves the DICOMs
    as filename: Accno_BodyPart/Side_View
    Base = 9833, Headers = 8245, images = 5458, header only fails = 0 (5458 saved)
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
            img, _, _, _, header = sdl.load_DICOM_2D(file)
        except Exception as e:
            #print('DICOM Error: %s' %e)
            continue

        # Retreive the view information. Only body part fail will fail us
        view, laterality, part, accno = 0, 0, 0, 0

        # Laterality: L, R or UK
        try: laterality = header['tags'].ImageLaterality.upper()
        except:
            try: laterality = header['tags'].Laterality.upper()
            except:
                try:
                    if 'LEFT' in header['tags'].StudyDescription.upper(): laterality = 'L'
                    else: laterality = 'R'
                except:
                    try:
                        if 'LEFT' in header['tags'].SeriesDescription.upper(): laterality = 'L'
                        else: laterality = 'R'
                    except Exception as e:
                        laterality = 'UKLAT'

        # Accession number
        try: accno = header['tags'].AccessionNumber
        except:
            try: accno = header['tags'].StudyID
            except:
                dir = os.path.dirname(file)
                accno = dir.split('/')[-3]

        # View 'LAT, PA, OBL, TAN. Maybe use SeriesNumber 1 = PA, 2 = LAT if 2 views etc
        try:
            view = header['tags'].ViewPosition.upper()
        except:
            try: accno = header['tags'].StudyID
            except Exception as e:
                view = 'UKVIEW'

        # PART: WRIST, HAND
        try:
            if 'WRIST' in header['tags'].StudyDescription.upper(): part = 'WRIST'
            elif 'HAND' in header['tags'].StudyDescription.upper(): part = 'HAND'
            elif 'WRIST' in header['tags'].SeriesDescription.upper(): part = 'WRIST'
            elif 'HAND' in header['tags'].SeriesDescription.upper(): part = 'HAND'
            else: part = 'UKPART'
        except:
            try: part = header['tags'].BodyPartExamined.upper()
            except Exception as e:
                print('Header error: %s' % e)
                continue

        """
            Some odd values that appear:
            view: TAN, LATERAL, NAVICULAR, LLO
            part: PORT WRIST, 
        # """

        # # Sometimes lateraity is off
        # if laterality != 'L' and laterality != 'R':
        #     if 'RIGHT' in header['tags'].StudyDescription.upper(): laterality = 'R'
        #     elif 'LEFT' in header['tags'].StudyDescription.upper(): laterality = 'L'
        #
        # # Set destination filename info
        # dst_File = accno + '_' + laterality + '-' + part + '_' + view
        #
        # # Skip non wrists or hands
        # if 'WRIST' not in part and 'HAND' not in part:
        #     print ('Skipping: ', dst_File)
        #     continue

        # Increment counters
        index += 1
        del img

    print('%s images saved (header)' % index)


def check_empty():

    """
    Check the raw CR subfolders for empty ones
    :return:
    """

    # First retreive lists of the the filenames
    folders = list()
    for (dirpath, dirnames, filenames) in os.walk(rawCR_dir):
        folders.append(filenames)

    # Load the filenames and randomly shuffle them
    save_path = home_dir + 'Cleaned_CR/'
    filenames = sdl.retreive_filelist('*', True, rawCR_dir)
    filenames2 = sdl.retreive_filelist('**', True, rawCR_dir)
    filenames += filenames2
    filenames3 = sdl.retreive_filelist('***', True, rawCR_dir)
    split = [x for x in filenames if len(x.split('/')) == 9]
    subsplit = [x for x in filenames if len(x.split('/')) == 8]
    print ('%s of %s Empty folders in %s' %(len(split), len(filenames), save_path))


check_empty()
process_raw()
#raw_to_PNG()
#check_raw()
