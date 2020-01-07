import bioformats as bf
# from xml.etree import ElementTree as ETree
from lxml import etree
import numpy as np
import os
from setup import copy_settings_default, settings_file
from configparser import ConfigParser
import SimpleITK as sitk
import pandas as pd
# import re
import logging
from tempfile import gettempdir
from numba import njit
import czifile as czi


logger = logging.getLogger("colocalizer.image_processing")


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
        logger.debug(f"Constructing image for {self.channel_ID}")
        image = sitk.GetImageFromArray(image)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(self.image)
        writer.Execute(image)

        del self.metadata['Color']


    def make_labelled_img(self, lower_thresh:float=0.0, minimum_obj_size:int=5, Xspace:float=1.0, Yspace:float=1.0, Zspace:float=1.0):
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.image)
        image = reader.Execute()
        image.SetSpacing([Xspace, Yspace, Zspace])
        thresh_img = image>lower_thresh
        thresh_img = sitk.ConnectedComponent(thresh_img)
        self.labelled = sitk.RelabelComponent(thresh_img, sortByObjectSize=True, minimumObjectSize=minimum_obj_size)
        self.labelled.SetSpacing([Xspace, Yspace, Zspace])
        self.shape_stats = sitk.LabelShapeStatisticsImageFilter()
        self.shape_stats.Execute(self.labelled)
        self.int_stats = sitk.LabelIntensityStatisticsImageFilter()
        self.int_stats.Execute(self.labelled, image)


    def measure_area(self, sizeX, sizeY, sizeZ):
        filt = sitk.LabelShapeStatisticsImageFilter()
        filt.Execute(self.binary)
        return filt.GetNumberOfPixels(1) * sizeX * sizeY * sizeZ


class ImageHandler(object):
    def __init__(self, metadata:dict):
        self.metadata = metadata
        self.metadata["ScalingX"] = float(self.metadata["ScalingX"]) * 1e+6
        self.metadata["ScalingY"] = float(self.metadata["ScalingY"]) * 1e+6
        self.metadata["ScalingZ"] = float(self.metadata["ScalingZ"]) * 1e+6
        self.channel_images = {}

    def add_channel_image(self, channel_image):
        self.channel_images[channel_image.channel_ID] = channel_image

    def make_colocalization_image(self):
        red = self.channel_images['Red'].labelled
        grn = self.channel_images['Green'].labelled
        filt = sitk.AndImageFilter()
        colocalizations = filt.Execute(red, grn)
        self.colocalizations = sitk.ConnectedComponent(colocalizations)
        self.colocalizations.SetSpacing([float(self.metadata["ScalingX"]),
                                         float(self.metadata["ScalingY"]),
                                         float(self.metadata["ScalingZ"])])
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
        marker_stats_filters = [self.channel_images[marker].shape_stats for marker in self.marker_names]
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


    def perform_spot_measurements(self):

        def get_ee_distances(distance_stats_filter, map:bool=False):
            labels_ = []
            edge_distances = []
            size_um = []
            size_pixels = []

            for label in distance_stats_filter.GetLabels():
                # Using minimum for each label gives us edge to edge distance
                if map:
                    labels_.append(self.colocalization_map[label].__str__())
                    edge_distances.append(distance_stats_filter.GetMinimum(label))
                    size_um.append(distance_stats_filter.GetPhysicalSize(label=label))
                    size_pixels.append(distance_stats_filter.GetNumberOfPixels(label=label))
                else:
                    labels_.append(label)
                    edge_distances.append(distance_stats_filter.GetMinimum(label))
                    size_um.append(distance_stats_filter.GetPhysicalSize(label=label))
                    size_pixels.append(distance_stats_filter.GetNumberOfPixels(label=label))
            # Construct dataframe
            df = pd.DataFrame(list(zip(labels_, edge_distances, size_pixels, size_um)),
                              columns=["Labels",
                                       'edge edge distance to DAPI [um]',
                                       "# Voxels",
                                       "Size [um]"])
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

        def get_actual_intensity_measurements(intensity_filter, map:bool=False):
            labels, intensities = zip(*[(label, intensity_filter.GetMean(label)) for label in
                      intensity_filter.GetLabels()])
            df = pd.DataFrame(list(zip(labels, intensities)),
                              columns=["Labels", 'Mean Intensity'])
            return df

        distance_map_from_all_nuclei = sitk.Abs(sitk.SignedMaurerDistanceMap(self.channel_images["Blue"].labelled,
                                                                             squaredDistance=False,
                                                                             useImageSpacing=True))
        self.dfs = {}
        for item in self.marker_names:
            int_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
            int_stats_filter.Execute(self.channel_images[item].labelled, distance_map_from_all_nuclei)
            logger.debug(f"Constucting edge to edge for {item}.")
            ee_distances = get_ee_distances(int_stats_filter)
            del int_stats_filter
            logger.debug(f"Constucting centroid to centroid for {item}.")
            cc_distances = get_cc_distances(self.channel_images[item].shape_stats)
            logger.debug(f"Constucting intensities for {item}.")
            intensities = get_actual_intensity_measurements(self.channel_images[item].int_stats, self.channel_images[item].shape_stats)
            logger.debug(f"Creating dataframes for {item}.")
            temp_df = pd.merge(ee_distances, cc_distances, on="Labels")
            self.dfs[item] = pd.merge(temp_df, intensities, on="Labels")
            del  ee_distances, cc_distances, intensities, temp_df
            logger.debug(f"Running dataframe calculations for {item}.")
            self.dfs[item]['Integrated Density'] = self.dfs[item]['Mean Intensity']/self.dfs[item]['# Voxels']

        int_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
        int_stats_filter.Execute(self.colocalizations, distance_map_from_all_nuclei)
        logger.debug(f"Constucting edge to edge for Colocalizations.")
        ee_distances = get_ee_distances(int_stats_filter, map=True)
        del int_stats_filter
        logger.debug(f"Constucting centroid to centroid for Colocalizations.")
        cc_distances = get_cc_distances(self.coloc_stats, map=True)
        logger.debug(f"Creating dataframes for Colocalizations.")
        self.dfs["Coloc"] = pd.merge(ee_distances, cc_distances, on="Labels")
        del ee_distances, cc_distances


def elem2dict(node):
    """
    Convert an lxml.etree node tree into a dict.
    """
    result = {}

    for element in node.iterchildren():
        # Remove namespace prefix
        key = element.tag.split('}')[1] if '}' in element.tag else element.tag

        # Process element as tree element if the inner XML contains non-whitespace content
        if element.text and element.text.strip():
            value = element.text
        else:
            value = elem2dict(element)
        if key in result:

            if type(result[key]) is list:
                result[key].append(value)
            else:
                tempvalue = result[key].copy()
                result[key] = [tempvalue, value]
        else:
            result[key] = value
    return result


# def get_main_metadata(mdroot) -> dict:
#     objective = dict(mdroot.find("Instrument").find('Objective').items())
#     del objective['ID']
#     pixels = dict(mdroot.find("Image").find("Pixels").items())
#     del pixels['ID']
#     return dict(**pixels, **objective)


def get_main_metadata(mdroot) -> dict:
    ac = elem2dict([item for item in mdroot.iter("AcquisitionModeSetup")][0])
    objective = elem2dict([obj for obj in [item for item in mdroot.iter("Instrument")][0].iter("Objective")][0])
    pixels = elem2dict([item for item in mdroot.iter("Image")][0])
    return dict(**ac, **pixels, **objective)


# def get_channel_metadata(mdroot:etree._Element, channel_number:int):
#     channels = mdroot.find('Image').find('Pixels').findall('Channel')
#     for channel in channels:
#         channel_dict = dict(channel.items())
#         if channel_dict['ID'].endswith(str(channel_number)):
#             # Set channel color based on color map.
#             channel_dict["Color"] = [color for color in color_map if channel_dict['Name'].lower() in color_map[color]][0].title()
#             return channel_dict
#     return None


def get_channel_metadata(mdroot:etree._Element, channel_number:int):
    channel = [obj for obj in [item for item in mdroot.iter("Channels")][0].iter("Channel")][channel_number]
    channel_dict = elem2dict(channel)
    # Set channel color based on color map.
    channel_dict["Color"] = [color for color in color_map if channel.get('Name').lower() in color_map[color]][0].title()
    return channel_dict


# def get_img(filename = "test_images/Image0001_deconvolution.zvi"):
#     logger.debug("Retrieving metadata...")
#     md = bf.get_omexml_metadata(filename)
#     logger.debug("Creating image reader...")
#     rdr = bf.ImageReader(filename)
#     mdroot = ETree.fromstring(re.sub(' xmlns="[^"]+"', '', md, count=1))
#     del md
#     main_metadata = get_main_metadata(mdroot)
#     files_3d = []
#     for t in range(int(main_metadata['SizeT'])):
#         logger.debug(f"Time dimension loop {t}...")
#         for c in range(int(main_metadata['SizeC'])):
#             logger.debug(f"Channel dimension loop {c}...")
#             image3d = np.empty([int(item) for item in
#                                 [main_metadata['SizeZ'],
#                                  main_metadata['SizeY'], main_metadata['SizeX']]])
#             for z in range(int(main_metadata['SizeZ'])):
#                 logger.debug(f"Z dimension loop...{z}")
#                 try:
#                     image3d[z] = rdr.read(c=c, z=z, t=t, rescale=False)
#                 except Exception as e:
#                     logger.debug("Error reading image: {}".format(e))
#             this_file = os.path.join(save_dir, f"Image3D_t{t}_c{c}.npy")
#             files_3d.append(this_file)
#             np.save(this_file, image3d, allow_pickle=True)
#             del image3d
#     logger.debug("Made it through image reading!")
#     rdr.close()
#     logger.debug("Creating image handler...")
#     img = ImageHandler(main_metadata)
#     for c in range(int(main_metadata['SizeC'])):
#         logger.debug(f"Channel loop {c}")
#         chan_metadata = get_channel_metadata(mdroot, c)
#         file_of_interest = os.path.join(save_dir, f"Image3D_t0_c{c}.npy")
#         # if statement for testing only remove for production
#         if os.path.exists(file_of_interest) and file_of_interest in files_3d:
#             new_img = ChannelImage(image=np.load(file_of_interest), metadata=chan_metadata)
#             # Skipping coords for blue during colocalization testing.
#             img.add_channel_image(new_img)
#         else:
#             logger.debug(f"File of interest {file_of_interest} not available")
#             continue
#     return img

def get_img(filename):
    logger.debug(f"Using czifile to open {filename}")
    czifile = czi.CziFile(filename)
    logger.debug("Retrieving metadata...")
    md = czifile.metadata()
    parser = etree.XMLParser(remove_blank_text=True)
    mdroot = etree.XML(md, parser=parser)
    del md
    main_metadata = get_main_metadata(mdroot)
    channels = [obj for obj in [item for item in mdroot.iter("Dimensions")][0].iter("Channels")][0]
    del mdroot
    image = czifile.asarray().squeeze()
    del czifile
    files_3d = []
    for c in range(int(main_metadata['SizeC'])):
        logger.debug(f"Channel dimension loop {c}...")
        image3d = image[c]
        this_file = os.path.join(save_dir, f"Image3D_c{c}.npy")
        files_3d.append(this_file)
        np.save(this_file, image3d, allow_pickle=True)
        del image3d
    logger.debug("Made it through image reading!")
    logger.debug("Creating image handler...")
    img = ImageHandler(main_metadata)
    for c in range(int(main_metadata['SizeC'])):
        logger.debug(f"Channel loop {c}")
        chan_metadata = get_channel_metadata(channels, c)
        file_of_interest = os.path.join(save_dir, f"Image3D_c{c}.npy")
        # if statement for testing only remove for production
        if os.path.exists(file_of_interest) and file_of_interest in files_3d:
            new_img = ChannelImage(image=np.load(file_of_interest), metadata=chan_metadata)
            # Skipping coords for blue during colocalization testing.
            img.add_channel_image(new_img)
        else:
            logger.debug(f"File of interest {file_of_interest} not available")
            continue
    return img

def run_main(filename:str, red_thresh:int, red_obj_min:int, grn_thresh:int, grn_obj_min:int, blu_thresh:int, blu_obj_min:int):
    image = get_img(filename)
    sizeX = float(image.metadata['ScalingX'])
    sizeY = float(image.metadata['ScalingY'])
    sizeZ = float(image.metadata['ScalingZ'])
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
    image.perform_spot_measurements()
    save_file = os.path.join(os.path.dirname(filename), "output", os.path.splitext(os.path.basename(filename))[0] + ".tif")
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    logger.debug(f"Attempting to save markers tif for {filename}")
    writer = pd.ExcelWriter(os.path.join(os.path.dirname(save_file), f'{os.path.splitext(os.path.basename(filename))[0]}.xlsx'), engine='xlsxwriter')
    # Write each dataframe to a different worksheet.
    logger.debug(f"Attempting to save dataframes to excel for {filename}")
    image.colocalization_information_df.to_excel(writer, sheet_name='Colocalization Info')
    image.dfs["Coloc"].to_excel(writer, sheet_name='Colocalization distances')
    image.dfs["Red"].to_excel(writer, sheet_name='Red distances')
    image.dfs["Green"].to_excel(writer, sheet_name='Green distances')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    del writer, image.dfs
    logger.debug("Attempting to ditch imagehandler to save memory in preparation for save.")
    img_red = image.channel_images["Red"].labelled
    del image.channel_images['Red']
    img_grn = image.channel_images["Green"].labelled
    del image.channel_images['Green']
    img_blu = image.channel_images["Blue"].labelled
    del image
    logger.debug(f"Attempting to save markers tif for {filename}")
    save_image(img_red, img_grn, img_blu, save_file)



def save_image(red_channel, grn_channel, blu_channel, output_file):
    red_channel = sitk.BinaryThreshold(red_channel, lowerThreshold=1.0, insideValue=1, outsideValue=0)
    grn_channel = sitk.BinaryThreshold(grn_channel, lowerThreshold=1.0, insideValue=1, outsideValue=0)
    blu_channel = sitk.BinaryThreshold(blu_channel, lowerThreshold=1.0, insideValue=1, outsideValue=0)
    new_image = sitk.Cast(sitk.Compose(red_channel*255, grn_channel*255, blu_channel*255), sitk.sitkVectorFloat32)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_file)
    writer.Execute(new_image)