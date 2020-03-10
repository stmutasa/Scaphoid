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
import json

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/Code/Datasets/Scaphoid/'

rawCR_dir = home_dir + 'Raw_Scaphoid_CR/'
test_loc_folder = home_dir + 'Test/'
cleanCR_folder = home_dir + 'Cleaned_CR_Single/'

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


def filter_DICOM(file, show_debug=False):
    """
    Loads the DICOM image, filters by what body part and view we want, then returns the info
    :return:
    """

    # Load the Dicom
    try:
        img, _, _, _, header = sdl.load_DICOM_2D(file)
    except Exception as e:
        if show_debug: print('DICOM Error: %s' %e)
        return

    # Retreive the view information. Only body part fail will fail us
    view, laterality, part, accno = 0, 0, 0, 0

    """
    Some odd values that appear:
    view: TAN, LATERAL, NAVICULAR, LLO
    part: PORT WRIST, 
    """

    # Laterality: L, R or UK
    try: laterality = header['tags'].ImageLaterality.upper()
    except:
        try:
            laterality = header['tags'].Laterality.upper()
        except:
            try:
                if 'LEFT' in header['tags'].StudyDescription.upper():
                    laterality = 'L'
                else:
                    laterality = 'R'
            except:
                try:
                    if 'LEFT' in header['tags'].SeriesDescription.upper():
                        laterality = 'L'
                    else:
                        laterality = 'R'
                except Exception as e:
                    if show_debug: print('Laterality Error: %s' % e)
                    laterality = 'UKLAT'

    # Accession number
    try: accno = header['tags'].AccessionNumber
    except:
        try:
            accno = header['tags'].StudyID
        except:
            dir = os.path.dirname(file)
            accno = dir.split('/')[-3]

    # View 'LAT, PA, OBL, TAN. Maybe use SeriesNumber 1 = PA, 2 = LAT if 2 views etc
    try: view = header['tags'].ViewPosition.upper()
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
            if show_debug: print('Header Error: %s' %e)
            return

    # Return everything
    return img, view, laterality, part, accno, header


def check_stats():

    """
    This function checks the stats for the bounding boxes and prints averages
    """

    # Load the files and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, cleanCR_folder)
    shuffle(filenames)
    gtboxes = sdl.load_CSV_Dict('filename', 'gtboxes.csv')

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0
    heights, widths, ratio = [], [], []
    nheights, nwidths = [], []
    iheights, iwidths = [], []

    for file in filenames:

        # Load the info
        image, view, laterality, part, accno, header = filter_DICOM(file)

        # Skip parts
        if 'HAND' in part or 'WRIST' in part: continue

        # Set destination filename info
        dst_File = accno + '_' + laterality + '-' + part + '_' + view

        # Now compare to the annotations folder
        try: annotations = gtboxes[dst_File+'.png']['region_shape_attributes']
        except: continue

        # Success, convert the string to a dictionary
        gtbox = json.loads(annotations)
        if not gtbox: continue

        # Normalize the gtbox, to [ymin, xmin, ymax, xmax, cny, cnx, height, width, origshapey, origshapex]
        shape = image.shape
        gtbox = np.asarray([gtbox['y'], gtbox['x'], gtbox['y'] + gtbox['height'], gtbox['x'] + gtbox['width'],
                            gtbox['y']+(gtbox['height']/2), gtbox['x']+(gtbox['width']/2), gtbox['height'], gtbox['width']])
        norm_gtbox = np.asarray([gtbox[0]/shape[0], gtbox[1]/shape[1],
                                 gtbox[2]/shape[0], gtbox[3]/shape[1],
                                 gtbox[4]/shape[0], gtbox[5]/shape[1],
                                 gtbox[6] / shape[0], gtbox[7] / shape[1],
                                 shape[0], shape[1]]).astype(np.float32)


        # Increment counters
        index += 1
        print ('%s of %s Done: %s' %(index, len(filenames), dst_File))
        nheights.append(norm_gtbox[6])
        nwidths.append(norm_gtbox[7])
        heights.append(gtbox[6])
        widths.append(gtbox[7])
        ratio.append(gtbox[6]/gtbox[7])
        iheights.append(image.shape[0])
        iwidths.append(image.shape[1])
        del image

    # Done with all patients
    heights, widths, ratio = np.asarray(heights), np.asarray(widths), np.asarray(ratio)
    iheights, iwidths = np.asarray(iheights), np.asarray(iwidths)
    nheights, nwidths = np.asarray(nheights), np.asarray(nwidths)
    print('%s bounding boxes. H/W AVG: %.3f/%.3f Max: %.3f/%.3f STD: %.3f/%.3f' % (
    index, np.average(heights), np.average(widths), np.max(heights), np.max(widths), np.std(heights), np.std(widths)))
    print('%s Norm bounding boxes. H/W AVG: %.3f/%.3f Max: %.3f/%.3f STD: %.3f/%.3f' % (
        index, np.average(nheights), np.average(nwidths), np.max(nheights), np.max(nwidths),  np.std(nheights), np.std(nwidths)))
    print('%s Images. H/W AVG: %.1f/%.1f Max: %.1f/%.1f STD: %.3f/%.3f' % (
        index, np.average(iheights), np.average(iwidths), np.max(iheights), np.max(iwidths),  np.std(iheights), np.std(iwidths)))
    print('%s Ratios. Max: %.2f, Min: %.2f, Avg: %.2f STD: %.3f' % (
        index, np.max(ratio), np.min(ratio), np.average(ratio),  np.std(ratio)))


def check_stats2():

    """
    This function checks the stats for the bounding boxes and prints averages
    """

    # Load the files and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, cleanCR_folder)
    gtboxes = sdl.load_CSV_Dict('filename', 'gtboxes.csv')

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0
    mean_IOUs, skipped = [], 0

    for file in filenames:

        # Load the info
        image, view, laterality, part, accno, header = filter_DICOM(file)

        # Skip parts
        #if 'HAND' in part or 'WRIST' in part: continue
        if 'WRIST' not in part: continue

        # Set destination filename info
        dst_File = accno + '_' + laterality + '-' + part + '_' + view

        # Now compare to the annotations folder
        try: annotations = gtboxes[dst_File+'.png']['region_shape_attributes']
        except: continue

        # Success, convert the string to a dictionary
        gtbox = json.loads(annotations)
        if not gtbox: continue

        # Make gtbox
        gtbox = np.asarray([gtbox['y'], gtbox['x'], gtbox['y'] + gtbox['height'], gtbox['x'] + gtbox['width'],
                            gtbox['y'] + (gtbox['height'] / 2), gtbox['x'] + (gtbox['width'] / 2), gtbox['height'],
                            gtbox['width']])

        # Make anchor boxes and calculate IOUs
        rat, ratSD = 1.06, 0.232*1.25
        scaSD = (19.5/125.8 + 24.18/122.2) / 2
        anchors = generate_anchors(image, [125.8, 122.2], 16, ratios=[rat-ratSD, rat, rat+ratSD], scales=[1-scaSD, 1.0, 1+scaSD])
        IOUs = _iou_calculate(anchors[:, :4], gtbox[:4])
        mean_IOUs.append(np.max(IOUs))
        if np.max(IOUs) <= 0.6:
            skipped += 1
            sdd.display_single_image(image)

        # Increment counters
        index += 1
        print ('%s of %s Done: %s' %(index, len(filenames), dst_File))
        del image, anchors, gtbox
        if index > 100: break

    # Done with all patients
    mean_IOUs = np.asarray(mean_IOUs)
    print('%s bounding boxes. IOU AVG: %.3f Max: %.3f STD: %.3f Skipped: %s' % (
    index, np.average(mean_IOUs), np.max(mean_IOUs), np.std(mean_IOUs), skipped))


def FL_Test():

    """
    Checking focal loss code
    :return:
    """

    LBL = [[0, 1, 0],[0, 0, 1], [1, 0, 0], [0, 1, 0],[0, 0, 1], [1, 0, 0]]
    LOG = [[-2, 2, 0], [2, -2, 1], [3, 2, 1], [2, -2, 0], [-2, 2, 0], [-1, 0, 1]]

    labels = tf.constant(LBL, tf.uint8)
    logits = tf.constant(LOG, tf.float32)

    # To prevent underflow errors
    eps = 1e-7

    # Make array of ones and multiply by alpha
    av = [0.25, 0.75, 1]
    alpha = tf.multiply(tf.cast(labels, tf.float32), tf.transpose(av))
    a_balance = tf.reduce_sum(alpha, axis=-1, keepdims=True)

    # Normalize the logits to class probabilities
    prob = tf.nn.softmax(logits, -1)

    # Returns True where the labels equal 1
    labels_eq_1 = tf.equal(labels, 1)

    # Where one hot label tensor is 1, return the softmax unmodified, else return 1-softmax
    prob_true = tf.where(labels_eq_1, prob, 1 - prob)

    # Calculate the modulating factor
    modulating_factor = (1.0 - prob_true) ** 2

    log_prob = tf.log(prob + eps)
    # Now calculate the loss: FL(pt) = −(1 − p)^γ * α * log(p)
    loss = -tf.reduce_sum(a_balance * modulating_factor * tf.cast(labels, tf.float32) * log_prob, -1)

    var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(var_init)

        outputs = sess.run([labels, logits, alpha, prob, labels_eq_1, a_balance, prob_true, modulating_factor, log_prob, loss])
        print (outputs)


def _iou_calculate(boxes1, boxes2):

    """
    Calculates the IOU of two boxes
    :param boxes1: [n, 4] [ymin, xmin, ymax, xmax]
    :param boxes2: [n, 4]
    :return: Overlaps of each box pair (aka DICE score)
    """

    # Split the coordinates
    ymin_1, xmin_1, ymax_1, xmax_1 = np.split(boxes1, indices_or_sections=len(boxes1[1]), axis=1)  # ymin_1 shape is [N, 1]..
    ymin_2, xmin_2, ymax_2, xmax_2 = np.split(boxes2, indices_or_sections=len(boxes2), axis=0)  # ymin_2 shape is [M, ]..

    # Retreive any overlaps of the corner points of the box
    max_xmin, max_ymin = np.maximum(xmin_1, xmin_2), np.maximum(ymin_1, ymin_2)
    min_xmax, min_ymax = np.minimum(xmax_1, xmax_2), np.minimum(ymax_1, ymax_2)

    # Retreive overlap along each dimension: Basically if the upper right corner of one box is above the lower left of another, there is overlap
    overlap_h = np.maximum(0., min_ymax - max_ymin)  # avoid h < 0
    overlap_w = np.maximum(0., min_xmax - max_xmin)

    # Cannot overlap if one of the dimension overlaps is 0
    overlaps = overlap_h * overlap_w

    # Calculate the area of each box
    area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
    area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

    # Calculate overlap (intersection) over union like Dice score. Union is just the areas added minus the overlap
    iou = overlaps / (area_1 + area_2 - overlaps)

    return iou


def generate_anchors(image, base_anchor_size, feature_stride, ratios, scales):

    """
    For generating anchor boxes
    :param image: The input image over which we generate anchors
    :param base_anchor_size: The base anchor size for this feature map
    :param feature_stride: int, stride of feature map relative to the image in pixels
    :param ratios: [1D array] of anchor ratios of width/height. i.e [0.5, 1, 2]
    :param scales: [1D array] of anchor scales in original space
    :return:
    """

    # Make Numpy Arrays
    if type(base_anchor_size) is not np.ndarray: base_anchor_size = np.asarray(base_anchor_size)
    if type(ratios) is not np.ndarray: ratios = np.asarray(ratios)
    if type(scales) is not np.ndarray: scales = np.asarray(scales)

    # Get shape
    shapey, shapex = image.shape

    # Generate a base anchor
    base_anchor = np.array([0, 0, base_anchor_size[0], base_anchor_size[1]], np.float32)
    base_anchors = _enum_ratios(_enum_scales(base_anchor, scales), ratios)
    _, _, ws, hs = np.split(base_anchors, indices_or_sections=len(base_anchors[1]), axis=1)

    # Create sequence of numbers
    y_centers = np.arange(0, shapey, feature_stride)
    x_centers = np.arange(0, shapex, feature_stride)

    # Broadcast parameters to a grid of x and y coordinates
    x_centers, y_centers = np.meshgrid(x_centers, y_centers)
    ws, x_centers = np.meshgrid(ws, x_centers)
    hs, y_centers = np.meshgrid(hs, y_centers)

    # Stack anchor centers and box sizes. Reshape to get a list of (x, y) and a list of (h, w)
    anchor_centers = np.reshape(np.stack([x_centers, y_centers], 2), [-1, 2])
    box_sizes = np.reshape(np.stack([ws, hs], axis=2), [-1, 2])

    # Convert to corner coordinates
    anchors = np.concatenate([anchor_centers - 0.5 * box_sizes, anchor_centers + 0.5 * box_sizes], axis=1)

    # Append all to one array with anchor [corner coordinates, centers, box_sizes]
    all = np.append(anchors, np.append(anchor_centers, box_sizes, axis=1), axis=1)

    # Filter outside anchors
    filtered = _filter_outside_anchors(anchors=all, img_dim=image.shape)

    return filtered


def _enum_scales(base_anchor, anchor_scales):

    '''
    :param base_anchor: [y_center, x_center, h, w]
    :param anchor_scales: different scales, like [0.5, 1., 2.0]
    :return: return base anchors in different scales.
            Example:[[0, 0, 128, 128],[0, 0, 256, 256],[0, 0, 512, 512]]
    '''

    anchor_scales = base_anchor * np.reshape(np.asarray(anchor_scales, dtype=np.float32), newshape=(len(anchor_scales), 1))
    return anchor_scales


def _enum_ratios(anchors, anchor_ratios):

    '''
    :param anchors: base anchors in different scales
    :param anchor_ratios:  ratio = h / w
    :return: base anchors in different scales and ratios
    '''

    # Unstack along the vertical dimension
    _, _, hs, ws = np.split(anchors, indices_or_sections=len(anchors[1]), axis=1)

    # Calculate squares of the anchor ratios
    sqrt_ratios = np.sqrt(anchor_ratios)
    sqrt_ratios = np.transpose(np.expand_dims(sqrt_ratios, axis=1))

    # Reshape the anchors
    ws = np.reshape(ws / sqrt_ratios, [-1])
    hs = np.reshape(hs * sqrt_ratios, [-1])

    num_anchors_per_location = ws.shape[0]

    ratios = np.transpose(np.stack([np.zeros([num_anchors_per_location, ]), np.zeros([num_anchors_per_location, ]), ws, hs]))

    return ratios


def _filter_outside_anchors(anchors, img_dim):

    """
    Removes anchor proposals with values outside the image
    :param anchors: The anchor proposals [xmin, ymin, xmax, ymax]
    :param img_dim: image dimensions (assumes square input)
    :return: the indices of the anchors not outside the image boundary
    """

    # Unpack the rank R tensor into multiple rank R-1 tensors along axis
    ymin, xmin, ymax, xmax, cny, cnx, wh, ww = np.split(anchors, indices_or_sections=len(anchors[1]), axis=1)

    # Return True for indices inside the image
    xmin_index, ymin_index = np.greater_equal(xmin, 0), np.greater_equal(ymin, 0)
    xmax_index, ymax_index = np.less_equal(xmax, img_dim[1]), np.less_equal(ymax, img_dim[0])

    # Now clean up the indices and return them
    indices = np.transpose(np.squeeze(np.stack([ymin_index, xmin_index, ymax_index, xmax_index])))
    indices = indices.astype(np.float32)
    indices = np.sum(indices, axis=1)
    indices = np.where(np.equal(indices, 4.0))

    # Gather the valid anchors
    valid_anchors = np.squeeze(np.take(anchors, indices, axis=0))

    return valid_anchors

#check_stats2()