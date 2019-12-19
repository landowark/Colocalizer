import bioformats as bf
from xml.etree import ElementTree as ETree
import numpy as np
import os
from setup import copy_settings_default, settings_file
from configparser import ConfigParser
import SimpleITK as sitk
import pandas as pd
import re
import logging
from tempfile import gettempdir
from numba import njit


logger = logging.getLogger("lamin_quant.image_processing")


config = ConfigParser()
try:
    config.read(settings_file)
    dyes = dict(config['DYES'].items())
except KeyError as e:
    logging.error(e)
    copy_settings_default()
    config.read(settings_file)
    dyes = dict(config['DYES'].items())
del dyes['last_dir']
color_map = {item : [thing.strip(" ").lower() for thing in dyes[item].split(",")] for item in dyes}


save_dir = gettempdir()
    # os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


@njit
def first_index(arr, val):
    '''
    Find the index of the first appearance of val in arr. The loop is extremely slow in Python, but we use
    the numba just-in-time compiler to optimize it (@njit decorator). Another option is to use np.where
    which returns all indexes but that is significantly (six times) slower than using the numba optimized function.
    '''
    for index, value in np.ndenumerate(arr):
        if val == value:
             return index



class ChannelImage(object):
    def __init__(self, image:np.ndarray, metadata:dict):
        self.metadata = metadata
        self.channel_ID = metadata['Color']
        del metadata
        self.image = os.path.join(save_dir, self.channel_ID + ".nrrd")
        if self.channel_ID == "Blue":
            # remove background
            image = image - (np.mean(image) + np.std(image))
            # floor values to zero
            image[image<0] = 0
        print(f"Constructing image for {self.channel_ID}")
        image = sitk.GetImageFromArray(image)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(self.image)
        writer.Execute(image)
        del self.metadata['ID']
        del self.metadata['Color']


    def make_labelled_img(self, lower_thresh:float=0.0, minimum_obj_size:int=5, Xspace:float=1.0, Yspace:float=1.0, Zspace:float=1.0):
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.image)
        image = reader.Execute()
        thresh_img = image>lower_thresh
        thresh_img = sitk.ConnectedComponent(thresh_img)
        self.labelled = sitk.RelabelComponent(thresh_img, sortByObjectSize=True, minimumObjectSize=minimum_obj_size)
        self.labelled.SetSpacing([Xspace, Yspace, Zspace])
        self.stats = sitk.LabelShapeStatisticsImageFilter()
        self.stats.Execute(self.labelled)


    def measure_area(self, sizeX, sizeY, sizeZ):
        filt = sitk.LabelShapeStatisticsImageFilter()
        filt.Execute(self.binary)
        return filt.GetNumberOfPixels(1) * sizeX * sizeY * sizeZ


class ImageHandler(object):
    def __init__(self, metadata:dict):
        self.metadata = metadata
        self.channel_images = {}

    def add_channel_image(self, channel_image):
        self.channel_images[channel_image.channel_ID] = channel_image

    def make_colocalization_image(self):
        red = self.channel_images['Red'].labelled
        grn = self.channel_images['Green'].labelled
        filt = sitk.AndImageFilter()
        colocalizations = filt.Execute(red, grn)
        self.colocalizations = sitk.ConnectedComponent(colocalizations)
        self.colocalizations.SetSpacing([float(self.metadata["PhysicalSizeX"]),float(self.metadata["PhysicalSizeY"]),float(self.metadata["PhysicalSizeZ"])])
        self.coloc_stats = sitk.LabelShapeStatisticsImageFilter()
        self.coloc_stats.Execute(self.colocalizations)
        self.colocalization_map = {}
        colocolization_labeled_segmentation_arr_view = sitk.GetArrayViewFromImage(self.colocalizations)
        for label in self.coloc_stats.GetLabels():
            # The index into the numpy array needs to be flipped as the order in numpy is zyx and in SimpleITK xyz
            index = first_index(colocolization_labeled_segmentation_arr_view, label)[::-1]
            self.colocalization_map[label] = [labeled_seg[index] for labeled_seg in
                                                          [self.channel_images["Green"].labelled,
                                                           self.channel_images["Red"].labelled]]

    def perform_colocalization_measurements(self):
        # Compute statistics for the colocalizations. Work with a list of lists and then
        # combine into a dataframe, faster than appending to the dataframe one by one.
        self.marker_names = [self.channel_images[channel].channel_ID for channel in self.channel_images if self.channel_images[channel].channel_ID != "Blue"]
        marker_stats_filters = [self.channel_images[marker].stats for marker in self.marker_names]
        column_titles = ['colocalization size'] * 2 + [item for sublist in [[marker] * 4 for marker in self.marker_names]
                                                       for item in sublist]
        all_colocalizations_data = []
        for item in self.colocalization_map.items():
            coloc_size = self.coloc_stats.GetPhysicalSize(item[0])
            marker_labels_list = item[1]
            current_colocalization = [coloc_size, self.coloc_stats.GetNumberOfPixels(item[0])] + \
                                     [item for sublist in [
                                         [label, filt.GetPhysicalSize(label), filt.GetNumberOfPixels(label),
                                          coloc_size / filt.GetPhysicalSize(label)]
                                         for label, filt in zip(marker_labels_list, marker_stats_filters)] for item in
                                      sublist]
            all_colocalizations_data.append(current_colocalization)

        self.colocalization_information_df = \
            pd.DataFrame(all_colocalizations_data, columns=column_titles)
        marker_columns = ['label', 'size [um^3]', 'size[voxels]', 'colocalization percentage']
        self.colocalization_information_df.columns = pd.MultiIndex.from_tuples(zip(self.colocalization_information_df.columns,
                                                                              ['um^3', 'voxels'] + [item for sublist in
                                                                                                    [marker_columns for
                                                                                                     item in
                                                                                                     self.marker_names] for
                                                                                                    item in sublist]))


    def perform_distance_measurements(self):

        def get_ee_distances(distance_stats_filter, map:bool=False):
            labels_ = []
            edge_distances = []
            for label in distance_stats_filter.GetLabels():
                # Using minimum for each label gives us edge to edge distance
                if map:
                    labels_.append(self.colocalization_map[label].__str__())
                    edge_distances.append(distance_stats_filter.GetMinimum(label))
                else:
                    labels_.append(label)
                    edge_distances.append(distance_stats_filter.GetMinimum(label))
            # Construct dataframe
            df = pd.DataFrame(list(zip(labels_, edge_distances)), columns=["Labels", 'edge edge distance to DAPI [um]'])
            return df

        def get_cc_distances(shape_stats_filter, map:bool=False):
            if map:
                labels, centroids = zip(*[(self.colocalization_map[label].__str__(), shape_stats_filter.GetCentroid(label)) for label in
                                          shape_stats_filter.GetLabels()])
            else:
                labels, centroids = zip(*[(label, shape_stats_filter.GetCentroid(label)) for label in
                                            shape_stats_filter.GetLabels()])
            centroids = np.array(centroids)
            dapi_stats_filter = sitk.LabelShapeStatisticsImageFilter()
            dapi_stats_filter.Execute(self.channel_images["Blue"].labelled)
            nuclei_labels, nuclei_centroids = zip(*[(nucleus, dapi_stats_filter.GetCentroid(nucleus)) for nucleus in dapi_stats_filter.GetLabels()])
            nuclei_centroids = np.array(nuclei_centroids)
            # Compute minimal distances and matching labels
            all_distances = -2 * np.dot(centroids, nuclei_centroids.T)
            all_distances += np.sum(centroids ** 2, axis=1)[:, np.newaxis]
            all_distances += np.sum(nuclei_centroids ** 2, axis=1)
            all_distances = np.sqrt(all_distances)

            min_indexes = np.argmin(all_distances, axis=1)
            # construct dataframe
            df = pd.DataFrame(list(zip(labels, all_distances[np.arange(len(min_indexes)), min_indexes])), columns=["Labels", 'centroid centroid distance [um]'])
            return df

        distance_map_from_all_nuclei = sitk.Abs(sitk.SignedMaurerDistanceMap(self.channel_images["Blue"].labelled,
                                                                             squaredDistance=False,
                                                                             useImageSpacing=True))
        self.dfs = {}
        for item in self.marker_names:
            int_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
            int_stats_filter.Execute(self.channel_images[item].labelled, distance_map_from_all_nuclei)
            ee_distances = get_ee_distances(int_stats_filter)
            del int_stats_filter
            sha_stats_filter = sitk.LabelShapeStatisticsImageFilter()
            sha_stats_filter.Execute(self.channel_images[item].labelled)
            cc_distances = get_cc_distances(sha_stats_filter)
            del sha_stats_filter
            self.dfs[item] = pd.merge(ee_distances, cc_distances, on="Labels")
            del  ee_distances, cc_distances

        int_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
        int_stats_filter.Execute(self.colocalizations, distance_map_from_all_nuclei)
        ee_distances = get_ee_distances(int_stats_filter, map=True)
        del int_stats_filter
        sha_stats_filter = sitk.LabelShapeStatisticsImageFilter()
        sha_stats_filter.Execute(self.colocalizations)
        cc_distances = get_cc_distances(sha_stats_filter, map=True)
        del sha_stats_filter
        self.dfs["Coloc"] = pd.merge(ee_distances, cc_distances, on="Labels")
        del ee_distances, cc_distances


def get_main_metadata(mdroot) -> dict:
    objective = dict(mdroot.find("Instrument").find('Objective').items())
    del objective['ID']
    pixels = dict(mdroot.find("Image").find("Pixels").items())
    del pixels['ID']
    return dict(**pixels, **objective)


def get_channel_metadata(mdroot:ETree.Element, channel_number:int):
    channels = mdroot.find('Image').find('Pixels').findall('Channel')
    for channel in channels:
        channel_dict = dict(channel.items())
        if channel_dict['ID'].endswith(str(channel_number)):
            # Set channel color based on color map.
            channel_dict["Color"] = [color for color in color_map if channel_dict['Name'].lower() in color_map[color]][0].title()
            return channel_dict
    return None


def get_img(filename = "test_images/Image0001_deconvolution.zvi"):
    print("Retrieving metadata...")
    md = bf.get_omexml_metadata(filename)
    print("Creating image reader...")
    rdr = bf.ImageReader(filename)
    mdroot = ETree.fromstring(re.sub(' xmlns="[^"]+"', '', md, count=1))
    del md
    main_metadata = get_main_metadata(mdroot)
    files_3d = []
    for t in range(int(main_metadata['SizeT'])):
        print(f"Time dimension loop {t}...")
        for c in range(int(main_metadata['SizeC'])):
            print(f"Channel dimension loop {c}...")
            image3d = np.empty([int(item) for item in
                                [main_metadata['SizeZ'],
                                 main_metadata['SizeY'], main_metadata['SizeX']]])
            for z in range(int(main_metadata['SizeZ'])):
                print(f"Z dimension loop...{z}")
                try:
                    image3d[z] = rdr.read(c=c, z=z, t=t, rescale=False)
                except Exception as e:
                    print("Error reading image: {}".format(e))
            this_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), f"Image3D_t{t}_c{c}.npy")
            files_3d.append(this_file)
            np.save(this_file, image3d, allow_pickle=True)
            del image3d
    print("Made it through image reading!")
    rdr.close()
    print("Creating image handler...")
    img = ImageHandler(main_metadata)
    for c in range(int(main_metadata['SizeC'])):
        print(f"Channel loop {c}")
        chan_metadata = get_channel_metadata(mdroot, c)
        file_of_interest = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), f"Image3D_t0_c{c}.npy")
        # if statement for testing only remove for production
        if os.path.exists(file_of_interest) and file_of_interest in files_3d:
            new_img = ChannelImage(image=np.load(file_of_interest), metadata=chan_metadata)
            # Skipping coords for blue during colocalization testing.
            img.add_channel_image(new_img)
        else:
            print(f"File of interest {file_of_interest} not available")
            continue
    # os.remove(file_of_interest)
    # javabridge.kill_vm()
    return img


def run_main(filename:str, red_thresh:int, red_obj_min:int, grn_thresh:int, grn_obj_min:int, blu_thresh:int, blu_obj_min:int):
    image = get_img(filename)
    sizeX = float(image.metadata['PhysicalSizeX'])
    sizeY = float(image.metadata['PhysicalSizeY'])
    sizeZ = float(image.metadata['PhysicalSizeZ'])
    for key in image.channel_images:
        img = image.channel_images[key]
        # img.make_otsu_img()
        if img.channel_ID == "Red":
            img.make_labelled_img(lower_thresh=red_thresh, minimum_obj_size=red_obj_min, Xspace=sizeX, Yspace=sizeY, Zspace=sizeZ)
        if img.channel_ID == "Green":
            img.make_labelled_img(lower_thresh=grn_thresh, minimum_obj_size=grn_obj_min, Xspace=sizeX, Yspace=sizeY, Zspace=sizeZ)
        if img.channel_ID == "Blue":
            img.make_labelled_img(lower_thresh=blu_thresh, minimum_obj_size=blu_obj_min, Xspace=sizeX, Yspace=sizeY, Zspace=sizeZ)
    image.make_colocalization_image()
    image.perform_colocalization_measurements()
    image.perform_distance_measurements()
    save_file = os.path.join(os.path.dirname(filename), "output", os.path.splitext(os.path.basename(filename))[0] + ".tif")
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    writer = pd.ExcelWriter(os.path.join(os.path.dirname(save_file), f'{os.path.splitext(os.path.basename(filename))[0]}.xlsx'), engine='xlsxwriter')
    # Write each dataframe to a different worksheet.
    image.colocalization_information_df.to_excel(writer, sheet_name='Colocalization Info')
    image.dfs["Coloc"].to_excel(writer, sheet_name='Colocalization distances')
    image.dfs["Red"].to_excel(writer, sheet_name='Red distances')
    image.dfs["Green"].to_excel(writer, sheet_name='Green distances')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    save_image(image.channel_images['Red'].labelled, image.channel_images['Green'].labelled, image.channel_images['Blue'].labelled, save_file)



def save_image(red_channel, grn_channel, blu_channel, output_file):
    output_dir = os.path.dirname(output_file)
    # red_channel = sitk.LabelMapToBinary(sitk.Image(grn_channel.GetSize(), sitk.sitkLabelUInt8))
    red_channel = sitk.BinaryThreshold(red_channel, lowerThreshold=1.0, insideValue=1, outsideValue=0)
    grn_channel = sitk.BinaryThreshold(grn_channel, lowerThreshold=1.0, insideValue=1, outsideValue=0)
    blu_channel = sitk.BinaryThreshold(blu_channel, lowerThreshold=1.0, insideValue=1, outsideValue=0)
    print(red_channel.GetSize(), grn_channel.GetSize(), blu_channel.GetSize())
    new_image = sitk.Cast(sitk.Compose(red_channel*255, grn_channel*255, blu_channel*255), sitk.sitkVectorFloat32)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_file)

    writer.Execute(new_image)