import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from simple_colors import *
from tqdm.notebook import tqdm

from PIL import ImageDraw

from utils.util import *
from utils.util_colors_drawing import *
from utils import util_colors_drawing
from utils.util_fig_display import *
from preprocessing import filter
from preprocessing import slide
from preprocessing import tiles
from const import OUTPUT_EXPO_NAME
#from python_files.const import *
# #from python_files.const_roi import *
# from stat_analysis.colocalisation_analysis import Coloc_analysis
from segmentation_classification import classification
import os 
from Qptiff_codes import qptiff_codes
def trace(func):
    def wrapper(*args, **kwargs):
        print(f"▶️  Entering: {func.__qualname__}")
        result = func(*args, **kwargs)
        print(f"✅ Exiting: {func.__qualname__}")
        return result
    return wrapper

class Roi():
    """
    Represents a region of interest (ROI) for fluorescence or whole slide images (WSI).
    The ROI is defined by its origin and end coordinates, and can be used to extract
    various information from the image, such as the tissue mask, cell masks, and original image.
    The ROI can also be used for colocalisation and cluster analysis.
    
    Usage:

    roi = Roi(dataset_config, slide_num, origin_row, origin_col, end_row, end_col, channels_of_interest=None)
    roi.get_img_original(channels_of_interest=None, save=False)
    roi.get_mask_tissue(save=False)
    roi.get_table_cells()
    roi.get_cell_masks()

    #visualisation 
    roi.display_img_original()
    roi.display_mask_tissue()
    roi.display_cell_masks(channels_of_interest=None, classified=True, wt_borders=True)
    img, mask = roi.get_img_mask_specific_cell(id_cell, display=True)
    
    Attributes:
    - dataset_config: a DatasetConfig object containing information about the dataset
    - img_name: the name of the image file
    - slide_num: the slide number of the image
    - channels_of_interest: a list of channels to extract from the image
    - origin_row: the row index of the top-left corner of the ROI
    - origin_col: the column index of the top-left corner of the ROI
    - end_row: the row index of the bottom-right corner of the ROI
    - end_col: the column index of the bottom-right corner of the ROI
    - path_roi: the path to the directory containing the ROI files
    - roi_shape: the shape of the ROI without borders
    - roi_w_borders_shape: the shape of the ROI with borders
    - group_for_comparison: a group identifier for comparison analysis
    - image_w_borders: the original image with borders
    - mask_tissue_w_borders: the tissue mask with borders
    - table_cells_w_borders: a table of cell information with borders
    - masks_cells_w_borders: a list of cell masks with borders
    - colocalisation_analysis: a Coloc_analysis object for colocalisation analysis
    - cluster_anlysis: a Cluster_analysis object for cluster analysis
    
    Methods:
    - get_mask_tissue(save=False): returns the tissue mask and saves it to file if save=True
    - save_mask_tissue(): saves the tissue mask to file
    - get_img_original(channels_of_interest=None, save=False): returns the original image and saves it to file if save=True
    - _get_img_original_fluorescence(channels_of_interest="all"): helper method to extract the original image from a fluorescence image
    - _get_rgb_wsi(): helper method to extract the original image from a WSI
    - get_table_cells(): returns a table of cell information
    - get_cell_masks(): returns a list of cell masks
    - display_img_original(): displays the original image
    - display_mask_tissue(): displays the tissue mask
    - display_cell_masks(channels_of_interest=None, classified=True, wt_borders=True): displays the cell masks
    - get_img_mask_specific_cell(id_cell, display=True): returns the image and mask for a specific cell
    
    
    Note : particular_roi = (row_origin, col_origin, row_end, col_end)
    """
    @trace
    def __init__(self,dataset_config,slide_num, origin_row, origin_col, end_row, end_col,channels_of_interest=None,entire_image=False): 
        #Basics 
        if entire_image: 
            n_row_n_col = slide.get_info_img_from_column_name(slide_num, column_name="slide_shape_in_tile", dataset_config=dataset_config)
            origin_row, origin_col, end_row, end_col = 1, 1, n_row_n_col[0], n_row_n_col[1]
        else :
            assert origin_row <= end_row and origin_col <= end_col, "origin_row <= end_row and origin_col <= end_col"

        self.dataset_config = dataset_config  
        self.img_name = slide_num # dataset_config.mapping_img_name[slide_num-1] MODIFIED_CODE_HERE_FOR_SIMULATION
        self.slide_num = slide_num
        self.channels_of_interest = channels_of_interest
        self.origin_row = origin_row
        self.origin_col = origin_col
        self.end_row = end_row
        self.end_col = end_col
        #Computed from the basic 
        self.path_roi = create_path_roi(dataset_config,slide_num, origin_row, origin_col, end_row, end_col,entire_image=entire_image)
        self.roi_shape = (dataset_config.tile_height * (self.end_row - self.origin_row + 1),
                      dataset_config.tile_width * (self.end_col - self.origin_col + 1))
        self.roi_w_borders_shape = (dataset_config.tile_height * (self.end_row - self.origin_row + 1) + 2*dataset_config.roi_border_size,
                      dataset_config.tile_width * (self.end_col - self.origin_col + 1) + 2*dataset_config.roi_border_size)
        self.group_for_comparison = None


        image_w_borders = None
        mask_tissue_w_borders = None
        table_cells_w_borders = None #contains both inside and border cells
        masks_cells_w_borders = None 
        statistics_roi = None 

        colocalisation_analysis = None
        cluster_anlysis  = None 
        # print(blue("Roi shape"),self.roi_shape,blue("Roi with borders shape :"), self.roi_w_borders_shape)
        
        
        
        # self.dataset_config.qptiff_dir = dataset_config.qptiff_dir
        # self.dataset_config.tsv_dir = dataset_config.tsv_dir
        # self.qptiff_path = None
        # self.tsv_path_old = None
        # self.tsv_path_nucleus = None
        # self.bc_code = None
        # self.if_code = None
        # self.slide_id = None
    
    # @trace
    # def get_qptiff_files(self):

    #     # -----------------------------------------------
    #     # 1. FIND THE QPTIFF IMAGE AND EXTRACT IDENTIFIERS
    #     # -----------------------------------------------
    #     qptiff_files = [
    #         f for f in os.listdir(self.config.qptiff_dir)
    #         if f.lower().endswith(".qptiff")
    #     ]
    #     if not qptiff_files:
    #         raise ValueError("No .qptiff files found")

    #     # parse naming pattern
    #     info_list = []
    #     for f in qptiff_files:
    #         parts = f.split("-")
    #         # IMMU | BC1 | 0213 | FIXT | 01 | IF1 | 01_#_xxxx.qptiff
    #         bc_code = parts[1]        # BC1
    #         slide_id = parts[2]       # 0213
    #         if_code = parts[5]        # IF1
    #         info_list.append((slide_id, bc_code, if_code, f))

    #     # sort → keep the FIRST file, ignore second duplicates
    #     info_list_sorted = sorted(info_list, key=lambda x: x[-1])
    #     slide_id, bc_code, if_code, filename = info_list_sorted[0]

    #     self.slide_id = slide_id
    #     self.bc_code = bc_code
    #     self.if_code = if_code
    #     self.qptiff_path = os.path.join(self.config.qptiff_dir, filename)

    #     print(f" Selected QPTIFF for slide {slide_id}:")
    #     print("   ", self.qptiff_path)

    #     # -----------------------------------------------
    #     # 2. SEARCH IN TSV DIRECTORY FOR RELATED FILES
    #     # -----------------------------------------------
    #     tsv_files = [
    #         f for f in os.listdir(self.config.tsv_dir)
    #         if (f.endswith(".tsv") or f.endswith(".tsv.gz"))
    #         and (bc_code in f)
    #         and (if_code in f)
    #         and (slide_id in f)
    #     ]

    #     if not tsv_files:
    #         raise ValueError(f"No TSV/TSV.GZ found for {bc_code}-{slide_id}-{if_code}")

    #     # -----------------------------------------------
    #     # 3. CHECK COLUMNS AND ASSIGN TO RIGHT VARIABLES
    #     # -----------------------------------------------
    #     for f in tsv_files:
    #         full_path = os.path.join(self.config.tsv_dir, f)

    #         # Read first 5000 lines max, to avoid large load
    #         if f.endswith(".gz"):
    #             reader = gzip.open(full_path, "rt")
    #         else:
    #             reader = open(full_path, "r")

    #         try:
    #             df_head = pd.read_csv(reader, sep="\t", nrows=20)
    #         finally:
    #             reader.close()

    #         cols = df_head.columns

    #         if "phenotype_norm" in cols and self.tsv_path_old is None:
    #             self.tsv_path_old = full_path
    #             print(f"✓ phenotype_norm TSV found: {full_path}")

    #         if "nucleus.x.px" in cols and self.tsv_path_nucleus is None:
    #             self.tsv_path_nucleus = full_path
    #             print(f"✓ nucleus.x.px TSV found: {full_path}")

    #         if self.tsv_path_old and self.tsv_path_nucleus:
    #             break

    #     # -----------------------------------------------
    #     # 4. ENSURE BOTH FILES FOUND OR RAISE ERROR
    #     # -----------------------------------------------
    #     if self.tsv_path_old is None:
    #         raise ValueError(f"No TSV with phenotype_norm for slide {slide_id}")

    #     if self.tsv_path_nucleus is None:
    #         raise ValueError(f"No TSV with nucleus.x.px for slide {slide_id}")

    #     print(" All required TSV files found successfully.")
    
    
    
    
    
    
    
    @trace
    def get_mask_tissue_original(self, save = False, false_mask = True):
        """
        Get the mask of the tissue
        Input : 
            - save : boolean, if True, save the mask of the tissue
        Output :    
            - mask_tissue : array, mask of the tissue
            - mask_tissue_w_borders : array, mask of the tissue with the borders of size dataset_config.roi_border_size 
        return : 

        """
        if self.dataset_config.data_type == "fluorescence":
            if self.dataset_config.consider_image_with_channels : 

                _, filtered_np_img_binary = slide.get_filter_image_result(self.slide_num,thumbnail = False,channel_number = -1,dataset_config=self.dataset_config)
            else : 
                _, filtered_np_img_binary = slide.get_filter_image_result(self.slide_num,thumbnail = False,channel_number = None,dataset_config=self.dataset_config)
        else : 
            _, filtered_np_img_binary = slide.get_filter_image_result(self.slide_num,thumbnail = False,channel_number = None,dataset_config=self.dataset_config)
        mask_downscaled = plt.imread(filtered_np_img_binary)
        # display_mask(mask_downscaled)
        y_origin = ((self.origin_col-1)*self.dataset_config.tile_width-self.dataset_config.roi_border_size)
        x_origin = (self.origin_row-1)*self.dataset_config.tile_height-self.dataset_config.roi_border_size
        height = (self.end_row - self.origin_row + 1)*self.dataset_config.tile_height + 2*self.dataset_config.roi_border_size
        width = (self.end_col - self.origin_col + 1)*self.dataset_config.tile_width + 2*self.dataset_config.roi_border_size
        
        x_origin_downscaled = int(x_origin/self.dataset_config.preprocessing_config.scale_factor)
        y_origin_downscaled = int(y_origin/self.dataset_config.preprocessing_config.scale_factor)
        height_downscaled = int(height/self.dataset_config.preprocessing_config.scale_factor)
        width_downscaled = int(width/self.dataset_config.preprocessing_config.scale_factor)
        
        mask_tissue_w_borders_downscaled = mask_downscaled[max(0,x_origin_downscaled):x_origin_downscaled+width_downscaled,max(0,y_origin_downscaled):y_origin_downscaled+height_downscaled]
        # print("x_origin_downscaled",x_origin_downscaled)
        # print("x_origin_downscaled+width_downscaled",x_origin_downscaled+width_downscaled)
        # print("y_origin_downscaled",y_origin_downscaled)
        # print("y_origin_downscaled+height_downscaled",y_origin_downscaled+height_downscaled)


        mask_downscaled_pil = Image.fromarray(mask_tissue_w_borders_downscaled)
        mask_pil = mask_downscaled_pil.resize((width,height),resample=Image.NEAREST)
        mask_tissue_w_borders = np.array(mask_pil).astype(bool)
        
        # display_mask(mask_tissue_w_borders)
        if false_mask : 
            self.mask_tissue_w_borders = np.ones(self.roi_w_borders_shape).astype(bool)
        else : 
            self.mask_tissue_w_borders = mask_tissue_w_borders
    @trace
    def get_mask_physiological_part(self, physiological_part, save = False):
        pass

    @trace
    def save_mask_tissue(self):
        """
        TODO
        Save the mask of the tissue
        """
        path_mask = os.path.join(self.path_roi,"mask_tissue.png")
        #path_mask= "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/masks/IMMU-BC1-1140-FIXT-01-IF1-01_#_0296bfe21068affb025b0bea8706cc98_mask.png"
        path_mask_w_borders = os.path.join(self.path_roi,"mask_tissue_w_borders.png")
        #path_mask_w_borders="/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/masks/IMMU-BC1-1140-FIXT-01-IF1-01_#_0296bfe21068affb025b0bea8706cc98_mask.png"
        if self.dataset_config.data_type == "fluorescence":
            if channels_of_interest == None :
                channels_of_interest = self.dataset_config.channels_of_interest
            self.image_w_borders = self._get_img_original_fluorescence(channels_of_interest=channels_of_interest)
        elif self.dataset_config.data_type == "wsi": 
            self.mask_tissue_w_borders = self._get_rgb_wsi()
        else :
            raise Exception("There is no code for this kind of data")
    @trace
    def get_img_original(self, channels_of_interest = None, save = False):
        """
        Get the original image
        Fluo : Done 
        WSI : TODO 
        """
        if self.dataset_config.data_type == "fluorescence":
            if self.dataset_config.consider_image_with_channels :
                self.image_w_borders = self._get_img_original_fluorescence()
            # commented for testing   
            #     if channels_of_interest == None :
            #         channels_of_interest = self.dataset_config.channels_of_interest
            #         self.image_w_borders = self._get_img_original_fluorescence(channels_of_interest=channels_of_interest)
            # else : 
            #     self.image_w_borders = self._get_rgb_wsi()
        elif self.dataset_config.data_type == "wsi": 
            self.image_w_borders = self._get_rgb_wsi()
        else : 
            raise Exception("There is no code for this kind of data")
            
        if save : 
            self.save_img_original()
    @trace
    def _get_img_original_fluorescence_original(self, channels_of_interest="all"):
        """ 
        Same as get_tile 
        """
        slide_num = self.slide_num
        dataset_config = self.dataset_config
        x_origin = (self.origin_col-1)*dataset_config.tile_width-dataset_config.roi_border_size
        y_origin = (self.origin_row-1)*dataset_config.tile_height-dataset_config.roi_border_size
        height = (self.end_row - self.origin_row + 1)*dataset_config.tile_height + 2*dataset_config.roi_border_size
        width = (self.end_col - self.origin_col + 1)*dataset_config.tile_width + 2*dataset_config.roi_border_size

        if channels_of_interest == None : 
            channels_of_interest = dataset_config.channels_cells_to_segment

        image_w_borders= tiles.get_roi_czi(dataset_config, slide_num, x_origin, y_origin, height, width, channels_of_interest )
        
        if hasattr(self, 'mask_tissue_w_borders'):
            for key in image_w_borders.keys():
                image_w_borders[key] = image_w_borders[key]*self.mask_tissue_w_borders[:,:] if key != "RGB" else image_w_borders[key]*self.mask_tissue_w_borders[:,:,np.newaxis]

        return image_w_borders
    
    
    @trace
    def _get_img_original_fluorescence(self, channels_of_interest=None):
        """
        Load the first 9 pages (0–8) from a QPTiff file and return them
        in a dictionary where keys are page numbers. Page 8 is also added
        under the key 'RGB' because it contains a 3-channel RGB image.
        """
        
        # 2 change : 1_ cropping directly when loading the image , there is a methode in tifffile library
        #2_ you can directly load based on the pyramidal level and not loading only first 7 page
        #3_RGB is not important
        import tifffile
        import numpy as np
        img_dict = {}
        # qptiff_path = self.qptiff_codes.qptiff_path
        qptiff_path = "/root/cloud-data/eu-ngs/U1086086/Ali_A/DeepCellMap/data/Immucan_F/IMMU-BC1-1140-FIXT-01-IF1-01_#_0296bfe21068affb025b0bea8706cc98.qptiff"
        with tifffile.TiffFile(qptiff_path) as tif:
            # Load pages 0 to 8
            for page_i in range(9):
                arr = tif.pages[page_i].asarray(out='memmap')
                img_dict[page_i] = arr

                # # If page 8 is RGB → also store under "RGB"
                if page_i == 8 and arr.ndim == 3 and arr.shape[-1] == 3:
                    img_dict["RGB"] = arr
                    
        # ROI limits
        x_min, x_max, y_min, y_max, _, _, _, _ = self._get_roi_w_borders_lims()

        ROI_dict = {}

        for key, img in img_dict.items():

            # Grayscale pages → shape (H, W)
            if img.ndim == 2:
                ROI_dict[key] = img[y_min:y_max, x_min:x_max]

            # # RGB page → shape (H, W, 3)
            elif img.ndim == 3:
                ROI_dict[key] = img[y_min:y_max, x_min:x_max, :] 
        self.image_w_borders = ROI_dict              

        return ROI_dict


    
    
    @trace
    def _get_rgb_wsi(self):
        """
        TODO
        Get the original image for WSI images


        Creer la liste des tiles avec au moins THRESH_MIN_TISSU% de tissue

        IMPORTANT : img type = np.float32 et range(0,1)
        """
        slide_num = self.slide_num
        dataset_config = self.dataset_config
        x_origin = (self.origin_col-1)*dataset_config.tile_width-dataset_config.roi_border_size
        y_origin = (self.origin_row-1)*dataset_config.tile_height-dataset_config.roi_border_size
        height = (self.end_row - self.origin_row + 1)*dataset_config.tile_height + 2*dataset_config.roi_border_size
        width= (self.end_col - self.origin_col + 1)*dataset_config.tile_width + 2*dataset_config.roi_border_size
        image_w_borders = tiles.get_roi_wsi(dataset_config, slide_num, x_origin, y_origin, width, height)
        
        if hasattr(self, 'mask_tissue_w_borders'):
            image_w_borders = image_w_borders*self.mask_tissue_w_borders[:,:,np.newaxis]
        
        return image_w_borders
    @trace
    def display_save_img_original(self,save = False,channels_of_interest = "all", figsize = (40,40)):
        """
        TODO
        Save the original image
        """
        if self.dataset_config.data_type == "fluorescence":

            if channels_of_interest == "all" : 
                channels_to_display = self.image_w_borders.keys()
            else : 
                channels_to_display = channels_of_interest
            for key, value in self.image_w_borders.items():
                # if key in channels_of_interest or key == "RGB" or channels_of_interest == "all" :
                if key in channels_to_display:
                    if save :
                        pathsave = find_path_last(self.path_roi, "original_img_channel_{}".format(key))
                    else :
                        pathsave = None
                    if key == "RGB" :
                        f = display_rgb(value, title = "",figsize=figsize, pathsave = pathsave)

                        # display_rgb(value, title = self.img_name + " channel " + str(key),figsize=figsize, pathsave = pathsave)
                    if key == "img_all_channels_and_more" :
                        f = display_rgb(value, title = self.img_name + " channel " + str(key), pathsave = pathsave)
                    # else : 
                    #     f = display_mask(value, title = self.img_name + " channel " + str(key),pathsave=pathsave)
        elif self.dataset_config.data_type == "wsi": 
            f = display_rgb(self.image_w_borders,title = self.img_name )
        else : 
            raise Exception("There is no code for this kind of data")
    @trace        
    def filter_cells_in_mask_tissue(self,table_cells_w_borders):
        """ Filter the cells that belong to the mask of the tissue"""
        verbose = False
        n_cells_before = table_cells_w_borders.shape[0]
        scale_factor = self.dataset_config.preprocessing_config.scale_factor
        # if self.dataset_config.data_type == "fluorescence":
        if self.dataset_config.consider_image_with_channels :
                
            _, filtered_np_img_binary = slide.get_filter_image_result(self.slide_num,thumbnail = False,channel_number = -1,dataset_config=self.dataset_config)
        else : 
            _, filtered_np_img_binary = slide.get_filter_image_result(self.slide_num,thumbnail = False,channel_number = None,dataset_config=self.dataset_config)
        mask_tissue_image = plt.imread(filtered_np_img_binary)

        for id_row, cell_row in table_cells_w_borders.iterrows():
            x_in_mask, y_in_mask = int(cell_row["x_img"]/scale_factor), int(cell_row["y_img"]/scale_factor)
            
            if mask_tissue_image[x_in_mask, y_in_mask] == 0:
                table_cells_w_borders.drop(id_row, inplace=True)
        n_cells_after = table_cells_w_borders.shape[0]
        print(blue("There was {} cells on the image but {} where outside tissue".format(n_cells_before,n_cells_before-n_cells_after))) if verbose else None 
        return table_cells_w_borders
    @trace
    def get_table_cells(self, channels_of_interest = None, save = False,filter_cells_from_several_channels=True):
        """
        Get the table of the cells

        #take from slide computation and post-process to create columns "x_roi, x_roi_w_borders, y_roi, y_roi_w_borders, in_border"
        If not computed for the entire slide, segment the cells and save masks before returning table of cells
        Input :
            - channels_of_interest : list of int, channels of interest
            - save : boolean, if True, save the table of cells
        Output :
            - table_cells_w_borders : dataframe, table of cells
        """
        if self.dataset_config.data_type == "fluorescence":
            if channels_of_interest == None :
                channels_of_interest = self.dataset_config.channels_of_interest
            #take from slide computation and post-process to create columns "x_roi, x_roi_w_borders, y_roi, y_roi_w_borders, in_border"
            table_cells_w_borders = self._get_table_cells_fluorescence(channels_of_interest,filter_cells_from_several_channels=filter_cells_from_several_channels)
        else :
            table_cells_w_borders = self._get_table_cells_wsi()
        
        if hasattr(self, 'mask_tissue_w_borders'):
            table_cells_w_borders = self.filter_cells_in_mask_tissue(table_cells_w_borders)
        self.table_cells_w_borders = table_cells_w_borders
        if save : 
            path_table_cells = os.path.join(self.path_roi,"table_cells.csv")
            self.table_cells_w_borders.to_csv(path_table_cells,sep = ";", index = False)




    #####################################"
    # new functions
    @trace
    def _get_table_cells_fluorescence_Ali(self,channels_of_interest = "all", filter_cells_from_several_channels = True):
        path_csv = "/root/cloud-data/eu-ngs/U1086086/Ali_A/CSV_concate/Full_Csv1.csv" 
        # self.
        table_cells = pd.read_csv(path_csv, sep=";")

        # === 2. Get ROI limits in global coordinates ===
        # x_min, x_max, y_min, y_max, *_ = self._get_roi_w_borders_lims()
        #Added new one , filtered to the ROI only
        # Convert tile indices (1-based typical) to pixel box (x:cols, y:rows)
        # x_min = (self.origin_col - 1) * self.dataset_config.tile_width
        # y_min = (self.origin_row - 1) * self.dataset_config.tile_height
        # x_max = self.end_col * self.dataset_config.tile_width
        # y_max = self.end_row * self.dataset_config.tile_height


        # x_min, x_max, y_min, y_max, xmin_borders, xmax_borders, ymin_borders, ymax_borders = self._get_roi_w_borders_lims()
        x_min = self.origin_col * 1024
        x_max = (self.end_col + 1) * 1024

        y_min = self.origin_row * 1024
        y_max = (self.end_row + 1) * 1024

        xmin_borders = x_min - self.dataset_config.roi_border_size
        xmax_borders = x_max + self.dataset_config.roi_border_size
        ymin_borders = y_min - self.dataset_config.roi_border_size
        ymax_borders = y_max + self.dataset_config.roi_border_size

        # === 3. Filter cells by nucleus global position ===
        table_cells_in_roi = table_cells[
            (table_cells["nucleus.x.px"] >= x_min) &
            (table_cells["nucleus.x.px"] < x_max) &
            (table_cells["nucleus.y.px"] >= y_min) &    
            (table_cells["nucleus.y.px"] < y_max)
        ].copy()

        table_cells_in_roi["within_roi"] = True  # optional column for tracking
        # table_cells_in_roi = table_cells_in_roi.reset_index(drop=True)

        print("shape",table_cells_in_roi.shape)
        
        #### Adding bordr parts
        

        table_cells_in_border = table_cells[(table_cells['nucleus.x.px'] >= ymin_borders) & (table_cells['nucleus.x.px'] < ymax_borders) & (table_cells['nucleus.y.px'] >= xmin_borders) & (table_cells['nucleus.y.px'] < xmax_borders)]     
        print("table_cells_in_border.shape after 1st filter",table_cells_in_border.shape)
        table_cells_in_border = table_cells_in_border[(table_cells_in_border['nucleus.x.px'] < y_min ) | (table_cells_in_border['nucleus.x.px'] > y_max ) | (table_cells_in_border['nucleus.y.px'] < x_min) | (table_cells_in_border['nucleus.y.px'] >x_max)]     
        print("table_cells_in_border.shape after 2nd filter",table_cells_in_border.shape)

        table_cells_in_border['within_roi'] = False
        table_cells_w_borders = pd.concat([table_cells_in_roi,table_cells_in_border])
        if filter_cells_from_several_channels : 
            for cell_name in list(self.dataset_config.cells_from_multiple_channels.keys()):
                table_cells_w_borders = table_cells_w_borders[table_cells_w_borders["used_to_build_{}".format(cell_name)] == False]
        table_cells_w_borders = table_cells_w_borders.reset_index(drop=True)
        

        
        table_cells_w_borders["x_roi_w_borders"] = (table_cells_w_borders["nucleus.y.px"] - ymin_borders)
        table_cells_w_borders["y_roi_w_borders"] = (table_cells_w_borders["nucleus.x.px"] - xmin_borders)

        
        # table_cells_w_borders["x_roi_w_borders"] = (table_cells_w_borders["tile_row"]-(self.origin_row))*self.dataset_config.tile_height + table_cells_w_borders["x_tile"]+self.dataset_config.roi_border_size
        # table_cells_w_borders["y_roi_w_borders"] = (table_cells_w_borders["tile_col"]-(self.origin_col))*self.dataset_config.tile_width + table_cells_w_borders["y_tile"]+self.dataset_config.roi_border_size
        # # x_roi_w_borders, y_roi_w_borders
        
                # === DEBUG: Visualize ROI + Borders + Cells ===
        import matplotlib.pyplot as plt
        import numpy as np

        # Determine size of ROI with borders
        roi_h = (self.end_row - self.origin_row + 1) * self.dataset_config.tile_height + 2*self.dataset_config.roi_border_size
        roi_w = (self.end_col - self.origin_col + 1) * self.dataset_config.tile_width + 2*self.dataset_config.roi_border_size

        print("ROI (with borders) size:", roi_h, roi_w)

        # Create empty canvas
        debug_canvas = np.zeros((roi_h, roi_w), dtype=np.uint8)

        # Plot cells (white pixels)
        x_coords = table_cells_w_borders["x_roi_w_borders"].astype(int)
        y_coords = table_cells_w_borders["y_roi_w_borders"].astype(int)

        # Clip in case anything is out of boundaries — important for debugging
        valid_mask = (x_coords >= 0) & (x_coords < roi_h) & (y_coords >= 0) & (y_coords < roi_w)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        debug_canvas[x_coords, y_coords] = 255

        plt.figure(figsize=(8, 8))
        plt.imshow(debug_canvas, cmap="gray")
        plt.title("DEBUG — ROI With Borders — Cell Positions")
        plt.axis("off")
        plt.show()

        # Print some statistics
        print("Total cells plotted:", len(x_coords))
        print("Cells dropped (out of ROI):", len(table_cells_w_borders) - len(x_coords))
        print("\n=== DEBUG COORDINATES ===")
        print("x_roi_w_borders min/max:", 
            table_cells_w_borders["x_roi_w_borders"].min(),
            table_cells_w_borders["x_roi_w_borders"].max())

        print("y_roi_w_borders min/max:", 
            table_cells_w_borders["y_roi_w_borders"].min(),
            table_cells_w_borders["y_roi_w_borders"].max())

        print("ROI dim (h,w):", roi_h, roi_w)

        
        return table_cells_w_borders

    @trace
    def _get_table_cells_fluorescence_3(self,channels_of_interest = "all", filter_cells_from_several_channels = True):
        """
        Get the table of the cells for fluorescence images

        Test if done on the entire image and do it on this particular ROI if not
        """
        # #Test if done for entire slide 
        # path_classified_slide = os.path.join(self.dataset_config.dir_classified_img, "slide_{}_cells.csv".format(self.slide_num))
        particular_roi = (self.origin_row, self.origin_col, self.end_row, self.end_col)
        path_classified_slide = classification.get_path_table_cells(self.slide_num,self.dataset_config,particular_roi = None)
        
        path_classified_roi = classification.get_path_table_cells(self.slide_num,self.dataset_config,particular_roi = particular_roi)
        if os.path.exists(path_classified_slide):
            table_cells = pd.read_csv(path_classified_slide,sep = ";")

        else : #If not, compute it on ROI only 
            particular_roi = (self.origin_row, self.origin_col, self.end_row, self.end_col)
            print("particular_roi",particular_roi)
            table_cells = classification.segment_classify_cells(self.slide_num, self.dataset_config,channels_cells_to_segment=channels_of_interest, particular_roi = particular_roi)
            table_cells.to_csv(path_classified_roi, sep = ";", index = False)

        x_min, x_max, y_min, y_max, xmin_borders, xmax_borders, ymin_borders, ymax_borders = self._get_roi_w_borders_lims()
        # print("x_min, x_max, y_min, y_max, xmin_borders, xmax_borders, ymin_borders, ymax_borders",x_min, x_max, y_min, y_max, xmin_borders, xmax_borders, ymin_borders, ymax_borders)

        # table_cells_in_roi = table_cells[(table_cells['x'] >= x_min) & (table_cells['x'] < x_max) & (table_cells['y'] >= y_min) & (table_cells['y'] < y_max)]
        # print("Table cell post filtering by xlim and y lim : ",table_cells_in_roi.shape)
        # table_cells_in_roi['within_roi'] = True
        # table_cells_in_border = table_cells[(table_cells['x'] >= xmin_borders) & (table_cells['x'] < xmax_borders) & (table_cells['y'] >= ymin_borders) & (table_cells['y'] < ymax_borders)]     
        # table_cells_in_border['within_roi'] = False
        # table_cells_w_borders = pd.concat([table_cells_in_roi,table_cells_in_border])
        # table_cells_w_borders = table_cells_w_borders[table_cells_w_borders["channel_number"].isin(channels_of_interest)]
        # table_cells_w_borders = table_cells_w_borders.reset_index(drop=True)
        # table_cells_w_borders["x_roi_w_borders"] = (table_cells_w_borders["row"]-(self.origin_row))*self.dataset_config.tile_height + table_cells_w_borders["x_tile"]+self.dataset_config.roi_border_size
        # table_cells_w_borders["y_roi_w_borders"] = (table_cells_w_borders["col"]-(self.origin_col))*self.dataset_config.tile_width + table_cells_w_borders["y_tile"]+self.dataset_config.roi_border_size
        # # x_roi_w_borders, y_roi_w_borders

        table_cells_in_roi = table_cells[(table_cells['x_img'] >= x_min) & (table_cells['x_img'] < x_max) & (table_cells['y_img'] >= y_min) & (table_cells['y_img'] < y_max)]
        # print("Table cell post filtering by xlim and y lim : ",table_cells_in_roi.shape)
        table_cells_in_roi['within_roi'] = True
        table_cells_in_border = table_cells[(table_cells['x_img'] >= xmin_borders) & (table_cells['x_img'] < xmax_borders) & (table_cells['y_img'] >= ymin_borders) & (table_cells['y_img'] < ymax_borders)]     
        # print("table_cells_in_border.shape after 1st filter",table_cells_in_border.shape)
        table_cells_in_border = table_cells_in_border[(table_cells_in_border['x_img'] < x_min) | (table_cells_in_border['x_img'] > x_max) | (table_cells_in_border['y_img'] < y_min) | (table_cells_in_border['y_img'] > y_max)]     
        # print("table_cells_in_border.shape after 2nd filter",table_cells_in_border.shape)

        table_cells_in_border['within_roi'] = False
        table_cells_w_borders = pd.concat([table_cells_in_roi,table_cells_in_border])
        if filter_cells_from_several_channels : 
            for cell_name in list(self.dataset_config.cells_from_multiple_channels.keys()):
                table_cells_w_borders = table_cells_w_borders[table_cells_w_borders["used_to_build_{}".format(cell_name)] == False]
        table_cells_w_borders = table_cells_w_borders.reset_index(drop=True)
        table_cells_w_borders["x_roi_w_borders"] = (table_cells_w_borders["tile_row"]-(self.origin_row))*self.dataset_config.tile_height + table_cells_w_borders["x_tile"]+self.dataset_config.roi_border_size
        table_cells_w_borders["y_roi_w_borders"] = (table_cells_w_borders["tile_col"]-(self.origin_col))*self.dataset_config.tile_width + table_cells_w_borders["y_tile"]+self.dataset_config.roi_border_size
        # x_roi_w_borders, y_roi_w_borders

        return  table_cells_w_borders
        # return  self._get_table_cells_fluorescence(channels_of_interest)
        # If not, compute it on ROI


        return table_cells_w_borders
    @trace
    def _get_roi_w_borders_lims(self):
        """
        Get the limits of the ROI with borders
        """
        dataset_config = self.dataset_config
        x_min = (self.origin_row-1)*self.dataset_config.tile_height
        x_max = self.end_row*self.dataset_config.tile_height
        y_min = (self.origin_col-1)*self.dataset_config.tile_width 
        y_max = self.end_col*self.dataset_config.tile_width 

        x_min_borders = (self.origin_row-1)*self.dataset_config.tile_height-dataset_config.roi_border_size 
        x_max_borders = self.end_row*self.dataset_config.tile_height + dataset_config.roi_border_size
        y_min_borders = (self.origin_col-1)*self.dataset_config.tile_width-dataset_config.roi_border_size
        y_max_borders = self.end_col*self.dataset_config.tile_width + dataset_config.roi_border_size 
        # ---- clamp coordinations so ROI never goes outside ----
        #H = self.image_shape_H
        #W = self.image_shape_W
        # temporary only for this image we set dimension below
        H = 51840
        W = 46080
        x_min = max(0, min(x_min, H))
        x_max = max(0, min(x_max, H))
        y_min = max(0, min(y_min, W))
        y_max = max(0, min(y_max, W))

        x_min_borders = max(0, min(x_min_borders, H))
        x_max_borders = max(0, min(x_max_borders, H))
        y_min_borders = max(0, min(y_min_borders, W))
        y_max_borders = max(0, min(y_max_borders, W))
            
        
        return x_min, x_max, y_min, y_max, x_min_borders, x_max_borders, y_min_borders, y_max_borders
    @trace
    def _get_table_cells_wsi(self):
        """
        Get the table of the cells for microglia images

        Test if done on the entire image and do it on this particular ROI if not
        """
        particular_roi = (self.origin_row, self.origin_col, self.end_row, self.end_col)
        path_classified_slide = classification.get_path_table_cells(self.slide_num,self.dataset_config,particular_roi = None)
        path_classified_roi = classification.get_path_table_cells(self.slide_num,self.dataset_config,particular_roi = particular_roi)
        if os.path.exists(path_classified_slide):
            table_cells = pd.read_csv(path_classified_slide,sep = ";")
            # print(table_cell_slide)
        else : #If not, compute it on ROI only 
            particular_roi = (self.origin_row, self.origin_col, self.end_row, self.end_col)
            table_cells = classification.segment_classify_cells(self.slide_num, self.dataset_config,particular_roi=particular_roi)
            table_cells.to_csv(path_classified_roi, sep = ";", index = False)
        # print("shape table_cells",table_cells.shape)
        x_min, x_max, y_min, y_max, xmin_borders, xmax_borders, ymin_borders, ymax_borders = self._get_roi_w_borders_lims()
        table_cells_in_roi = table_cells[(table_cells['x_img'] >= x_min) & (table_cells['x_img'] < x_max) & (table_cells['y_img'] >= y_min) & (table_cells['y_img'] < y_max)]
        # print("Table cell post filtering by xlim and y lim : ",table_cells_in_roi.shape)
        table_cells_in_roi['within_roi'] = True
        table_cells_in_border = table_cells[(table_cells['x_img'] >= xmin_borders) & (table_cells['x_img'] < xmax_borders) & (table_cells['y_img'] >= ymin_borders) & (table_cells['y_img'] < ymax_borders)]     
        # print("table_cells_in_border.shape after 1st filter",table_cells_in_border.shape)
        table_cells_in_border = table_cells_in_border[(table_cells_in_border['x_img'] < x_min) | (table_cells_in_border['x_img'] > x_max) | (table_cells_in_border['y_img'] < y_min) | (table_cells_in_border['y_img'] > y_max)]     
        # print("table_cells_in_border.shape after 2nd filter",table_cells_in_border.shape)

        table_cells_in_border['within_roi'] = False
        table_cells_w_borders = pd.concat([table_cells_in_roi,table_cells_in_border])
        table_cells_w_borders = table_cells_w_borders.reset_index(drop=True)
        table_cells_w_borders["x_roi_w_borders"] = (table_cells_w_borders["tile_row"]-(self.origin_row))*self.dataset_config.tile_height + table_cells_w_borders["x_tile"]+self.dataset_config.roi_border_size
        table_cells_w_borders["y_roi_w_borders"] = (table_cells_w_borders["tile_col"]-(self.origin_col))*self.dataset_config.tile_width + table_cells_w_borders["y_tile"]+self.dataset_config.roi_border_size
        # x_roi_w_borders, y_roi_w_borders
        return  table_cells_w_borders
    @trace
    def get_cell_masks(self,channels_of_interest = None):
        """
        Get the mask of the cells
        Test if done on the entire image and do it on this particular ROI if not
        """
        if self.dataset_config.data_type == "fluorescence":
            if channels_of_interest == None :
                channels_of_interest = self.dataset_config.channels_of_interest
            self.masks_cells_w_borders = self._get_cell_masks_fluorescence(channels_of_interest=channels_of_interest)
        else : #microglia
            self.masks_cells_w_borders = self._get_cell_masks_wsi()
    @trace
    def _add_mask_cell_in_roi_mask(self,mask_roi, mask_cell,row, col, x_tile, y_tile, length_max, cell_type = None):
        """
        Add the mask of the cell in the mask of the ROI
        """
        if self.dataset_config.data_type == "wsi":
            mask_cell*=cell_type
            mask_cell = mask_cell.astype(np.uint8)
        length_max = max(256,length_max)
        # print("length_max",length_max)
        roi_border_size = self.dataset_config.roi_border_size
        tile_height = self.dataset_config.tile_height
        tile_width = self.dataset_config.tile_width

        x_in_mask_roi = roi_border_size + (row-self.origin_row)*tile_height + x_tile 
        y_in_mask_roi = roi_border_size + (col-self.origin_col)*tile_width + y_tile

        pad = int(length_max/2)
        mask_roi_padded = np.pad(mask_roi, ((pad,pad),(pad,pad)),"constant" ) #0 padding
        x_in_mask_roi+=pad
        y_in_mask_roi+=pad
        mask_roi_padded[x_in_mask_roi-int(length_max/2):x_in_mask_roi+int(length_max/2),y_in_mask_roi-int(length_max/2):y_in_mask_roi+int(length_max/2)] += mask_cell
        mask_roi = mask_roi_padded[pad:-pad,pad:-pad]

        return mask_roi


#### ############################
#New Function


    @trace
    def get_mask_tissue_CJ(self, af_channel=7, min_obj_size=500, hole_size=500, show_plots=True):
        """
        Build aligned tissue mask directly from the QPTIFF autofluorescence channel.
        Produces a mask matching DeepCellMap ROI with borders.
        """

        import tifffile
        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.filters import threshold_otsu
        from skimage.morphology import remove_small_holes, remove_small_objects

        # ===============================
        # 1) Read FULL-RES autofluorescence from QPTIFF
        # ===============================
        qptiff_path = self.qptiff_codes.qptiff_path
        print("🔵 Loading QPTIFF:", qptiff_path)

        with tifffile.TiffFile(qptiff_path) as tif:
            af_img = tif.pages[af_channel].asarray()

        H, W = af_img.shape
        print(f"AF channel shape: {af_img.shape}")

        # ===============================
        # 2) Normalize
        # ===============================
        af = af_img.astype(np.float32)
        af = (af - af.min()) / (af.max() - af.min() + 1e-8)

        # ===============================
        # 3) Otsu to detect tissue
        # ===============================
        print("🔵 Applying Otsu threshold…")
        t = threshold_otsu(af)
        mask_full = af > t

        # Clean mask
        mask_full = remove_small_objects(mask_full, min_size=min_obj_size)
        mask_full = remove_small_holes(mask_full, area_threshold=hole_size)

        mask_full = mask_full.astype(bool)
        print("Full mask created, shape:", mask_full.shape)

        # ===============================
        # 4) Plot full mask for validation
        # ===============================
        if show_plots:
            plt.figure(figsize=(7,7))
            plt.imshow(mask_full, cmap="gray")
            plt.title("FULL QPTIFF Tissue Mask (bool)")
            plt.axis("off")
            plt.show()

        # ===============================
        # 5) Crop ROI (DeepCellMap coordinates)
        # ===============================
        x_min, x_max, y_min, y_max, _,_,_,_ = self._get_roi_w_borders_lims()

        print("🔵 ROI boundaries:")
        print("y_min,y_max:", y_min, y_max)
        print("x_min,x_max:", x_min, x_max)

        mask_roi = mask_full[y_min:y_max, x_min:x_max]
        print("ROI mask shape:", mask_roi.shape)

        # ===============================
        # 6) Plot ROI mask
        # ===============================
        if show_plots:
            plt.figure(figsize=(7,7))
            plt.imshow(mask_roi, cmap="gray")
            plt.title("ROI Tissue Mask (with borders)")
            plt.axis("off")
            plt.show()

        # ===============================
        # 7) Store and return
        # ===============================
        self.mask_tissue_w_borders = mask_roi

        print("✅ Tissue mask stored in self.mask_tissue_w_borders")

        # Extra debugging
        print("Mask dtype:", mask_roi.dtype)
        print("Mask unique values:", np.unique(mask_roi))

        return mask_roi

        
    
    

    @trace
    def get_mask_tissue(self):
        
        from PIL import Image
        import os
        import h5py
        import pandas as pd
        import os
        import numpy as np

        cancer_type = "Breast"
        cohort = "BC1" #(should be in ["BC1", "BC2", "BC3", BC1_SYNERGY"])
        modality = "IF"
        pannel = "IF1" #(should be in ["IF1,"IF2","IF3"])
        patch_size = 256
        step_size = 256
        #info_image_path = ""
        slide_id = "IMMU-BC1-1140-FIXT-01-IF1-01_#_0296bfe21068affb025b0bea8706cc98" 


        patch_save_dir = os.path.join("/root/cloud-data/ts-eu/Theo_Perochon/outputs","processed_data",cancer_type,cohort,modality,pannel,"1_pre_processing","tissue_extraction_and_patching","patches",f"patch_size_{patch_size}_step_size_{step_size}",f"{slide_id}.h5")
        
        info_image_path = os.path.join("/root/cloud-data/ts-eu/Theo_Perochon/outputs","processed_data",cancer_type,cohort,modality,pannel,"0_images_info",f"{slide_id}.csv") # usefull to immediately get the image height and width to reconstruct the mask.

        path_img_info ='/root/cloud-data/ts-eu/Theo_Perochon/outputs/processed_data/Breast/BC1/IF/IF1/0_images_info/IMMU-BC1-1140-FIXT-01-IF1-01_#_0296bfe21068affb025b0bea8706cc98.csv'
        
        image_info = pd.read_csv(path_img_info)
        original_height = int(image_info.loc[0, "original_height"])
        original_width  = int(image_info.loc[0, "original_width"])
        coordinate_tissue_patches = os.path.join("/root/cloud-data/ts-eu/Theo_Perochon/outputs","processed_data",cancer_type,cohort,modality,pannel,"1_pre_processing","tissue_extraction_and_patching","patches",f"patch_size_{patch_size}_step_size_{step_size}",f"{slide_id}.h5")
        
        with h5py.File(coordinate_tissue_patches, "r") as f_coords:
            coords = f_coords['coords'][:] #contains [x,y] coordinates of top-left corner of each tissue patch of size 256
            
        #Example to create the mask for 1 image :
        mask = np.zeros((original_height, original_width), dtype=np.uint8)
        for i in range(len(coords)):
            x_beginning = coords[i][0]
            y_beginning = coords[i][1]
            mask[y_beginning:y_beginning+patch_size, x_beginning:x_beginning+patch_size] = 1
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.imshow(mask, cmap="gray")
        plt.title("Tissue Mask (1=tissue)")
        plt.axis("off")
        plt.show()    
        
        x_min, x_max, y_min, y_max,_,_,_,_ = self._get_roi_w_borders_lims()
        y0 = y_min
        x0 = x_min
        x1 = x_max
        y1 = y_max
        # # Convert tile indices (1-based typical) to pixel box (x:cols, y:rows)
        # x0 = (self.origin_col - 1) * self.dataset_config.tile_width
        # y0 = (self.origin_row - 1) * self.dataset_config.tile_height
        # x1 = self.end_col * self.dataset_config.tile_width
        # y1 = self.end_row * self.dataset_config.tile_height

        # Optional padding
        # x0 = max(0, x0 - pad_px)
        # y0 = max(0, y0 - pad_px)
        # x1 = min(w, x1 + pad_px)
        # y1 = min(h, y1 + pad_px)

        # Crop (rows then cols)
        mask = mask[y0:y1, x0:x1]
        print("ROI mask shape:", mask)
            
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.imshow(mask, cmap="gray")
        plt.title("Tissue Mask ROI (1=tissue)")
        plt.axis("off")
        plt.show()    
        
        # Store + optionally save
        self.mask_tissue_w_borders = mask 
        
        
        # Path where DeepCellMap expects masks (adjust to your dataset)
        out_dir = "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/masks"
        os.makedirs(out_dir, exist_ok=True)

        # Save as PNG (0/1 mask)
        out_path = os.path.join(out_dir, f"{slide_id}_mask.png")
        Image.fromarray((mask * 255).astype("uint8")).save(out_path)

        print(f"[OK] Mask saved at: {out_path}")


    @trace 
    def get_mask_tissue_OLD_2(self,save = False):
        """
        Get the mask of the tissue from a QPTIFF file
        Input : 
            - qptiff_path : str, path to the QPTIFF file
            - save : boolean, if True, save the mask of the tissue
        
        """
        import tifffile
        from PIL import Image
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
    

        # ---- Get image size from QPTIFF header only (no pixel read) ----
        qptiff_path = self.qptiff_codes.qptiff_path
        with tifffile.TiffFile(qptiff_path) as tif:
            # Using the first page dimensions
            page0 = tif.pages[0]
            # TIFF pages are (rows, cols) == (height, width)
            height, width = page0.shape
            print(f"Image size from QPTIFF: width={width}, height={height}")

        # read the csv , making a mask out of csv of the cells
         # ---- Read CSV ----
        info_csv = "/root/cloud-data/eu-ngs/U1086086/Ali_A/CSV_concate/Full_Csv1.csv"
        df = pd.read_csv(info_csv, sep=";")
        print("CSV loaded, number of rows:", len(df))
        print(df.columns.tolist()  )
        required_cols = {"nucleus.x.px", "nucleus.y.px"}
        if not required_cols.issubset(df.columns):
            missing_cols = list(required_cols - set(df.columns))
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        # Create an empty mask
        mask_tissue = np.zeros((height, width), dtype=np.uint8) 
        for index, row in df.iterrows():
            x = int(row["nucleus.x.px"])
            y = int(row["nucleus.y.px"])
            mask_tissue[y, x] = 1  # Mark the cell position in the mask
            
        # display_mask(self.mask_tissue)
        mask_img = (mask_tissue.astype(np.uint8)) * 255
        plt.figure(figsize=(6, 6))
        plt.imshow(mask_img[::150, ::150], cmap="gray")
        plt.title(f"Mask of the 'tissue'")
        plt.axis("off")
        plt.show()   
        #saving the mask of the whole tissue 
        # if save :   
        #     path_mask = "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/masks/mask_tissue_qptiff.png"
        #     mask_tissue_pil = Image.fromarray((self.mask_tissue*255).astype(np.uint8))
        #     mask_tissue_pil.save(path_mask)    
        #     self.mask_tissue = mask_tissue.astype(bool)

        #if not hasattr(self, "mask_tissue"):
        #   raise RuntimeError("self.mask_tissue is missing. Build it first (e.g., with get_mask_tissue_qptiff).")

        mask = mask_tissue
        h, w = mask.shape

        # Convert tile indices (1-based typical) to pixel box (x:cols, y:rows)
        x0 = (self.origin_col - 1) * self.dataset_config.tile_width
        y0 = (self.origin_row - 1) * self.dataset_config.tile_height
        x1 = self.end_col * self.dataset_config.tile_width
        y1 = self.end_row * self.dataset_config.tile_height

        # Optional padding
        # x0 = max(0, x0 - pad_px)
        # y0 = max(0, y0 - pad_px)
        # x1 = min(w, x1 + pad_px)
        # y1 = min(h, y1 + pad_px)

        # Crop (rows then cols)
        roi_mask = mask[y0:y1, x0:x1]
        print("ROI mask shape:", roi_mask.shape)
        
        
        # display_mask(self.mask_tissue) ROI
        mask_img = (roi_mask.astype(np.uint8)) * 255
        plt.figure(figsize=(6, 6))
        plt.imshow(mask_img[::25, ::25], cmap="gray")
        plt.title(f"Mask of the 'ROI Tissue'")
        plt.axis("off")
        plt.show()   
        
        
        # Store + optionally save
        self.mask_tissue_w_borders = roi_mask
        if save:
            out = f"/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/masks/mask_tissue_crop_r{self.origin_row}-{self.end_row}_c{self.origin_col}-{self.end_col}.png"
            Image.fromarray(roi_mask.astype(np.uint8) * 255).save(out)

        


    




            
        
        


    @trace
    def Calculate_mask_cell_qptiff(self,
        qptiff_path,
        info_csv,
        out_mask_path=None,
        label_names=None,
        label_priority=None,
        show_preview=False,
    ):
        """
        Rasterize per-cell circular masks (by cell area) into a single label image.

        Parameters
        ----------
        qptiff_path : str
            Path to the QPTIFF used ONLY to get width/height of the field.
        info_csv : str
            CSV containing at least: 'nucleus.x.px', 'nucleus.y.px', 'cell.area.px'
            and one-hot columns for cell types (e.g., 'B', 'T', etc.).
        out_mask_path : str or None
            If provided, saves an 8-bit PNG with label indices.
        label_names : dict[int,str] or None
            Mapping from integer code -> human-readable name.
            Default (your mapping) used if None.
        label_priority : list[str] or None
            Names (matching the one-hot column names) defining draw/overwrite order.
            Last drawn has highest priority where overlaps occur.
            If None, uses the order from label_names ascending by code.
        show_preview : bool
            If True, shows a Plotly preview of the label mask.

        Returns
        -------
        result : dict[str, np.ndarray]
            Per-class boolean masks (H, W).
        label_img : np.ndarray
            2D uint8 array (H, W) with integer labels (0 == background).
        counts : dict[str, int]
            Pixel counts per class name.
        """
        import math
        import numpy as np
        import pandas as pd
        import tifffile
        from PIL import Image, ImageDraw
        # Optional preview
        try:
            import plotly.express as px
        except Exception:
            px = None
        # ---- Defaults ----
        if label_names is None:
            label_names = {
                1: "B",
                2: "BnT",
                3: "DC",
                4: "MacCD163",
                5: "Neutrophil",
                6: "T",
                7: "Tumor",
                8: "Tumor_CD15",
                9: "other",
            }
        self.mapping_cells_colors = {
                            "other":        [127, 127, 127],  # gray
                            "Tumor":        [214, 39,  40],   # red
                            "MacCD163":     [255, 127, 14],   # orange
                            "T":            [44,  160, 44],   # green
                            "Tumor_CD15":   [148, 103, 189],  # purple
                            "Neutrophil":   [23,  190, 207],  # cyan
                            "DC":           [31,  119, 180],  # blue
                            "B":            [188, 189, 34],   # olive
                            "BnT":          [140, 86,  75]    # brown
        }
        print("gray:other , Tumor:red , MacCD163:orange , T:green , Tumor_CD15:purple , Neutrophil:cyan , DC:blue , B:olive , BnT:brown")
        # Validate mapping codes are small positive ints (we’ll store in uint8)
        if not all(isinstance(k, int) and 0 < k < 256 for k in label_names.keys()):
            raise ValueError("label_names keys must be 1..255")

        # If no priority specified, draw ascending by numeric label code
        # (later labels overwrite earlier ones).
        if label_priority is None:
            # translate codes -> names in ascending order
            label_priority = [label_names[k] for k in sorted(label_names.keys())]
        else:
            # Ensure user-provided priority names exist
            missing = [n for n in label_priority if n not in set(label_names.values())]
            if missing:
                raise ValueError(f"Names in label_priority not in label_names: {missing}")

        # ---- Get image size from QPTIFF header only (no pixel read) ----
        with tifffile.TiffFile(qptiff_path) as tif:
            # Using the first page dimensions
            page0 = tif.pages[0]
            # TIFF pages are (rows, cols) == (height, width)
            height, width = page0.shape
            print(f"Image size from QPTIFF: width={width}, height={height}")

        # ---- Read CSV ----
        df = pd.read_csv(info_csv, sep=";")
        print("CSV loaded, number of rows:", len(df))
        print(df.columns.tolist()  )
        required_cols = {"nucleus.x.px", "nucleus.y.px", "cell.area.px"}
        if not required_cols.issubset(df.columns):
            missing_cols = list(required_cols - set(df.columns))
            raise ValueError(f"CSV missing required columns: {missing_cols}")

        # Identify one-hot class columns from the provided label names
        class_cols = list(set(label_names.values()))
        missing_one_hot = [c for c in class_cols if c not in df.columns]
        if missing_one_hot:
            raise ValueError(f"CSV missing one-hot columns for classes: {missing_one_hot}")

        # Keep only valid rows (non-NaN positions and non-negative area)
        base_mask = df["nucleus.x.px"].notna() & df["nucleus.y.px"].notna() & df["cell.area.px"].notna()
        # Filter out zero/negative area to avoid r = 0 or NaN
        base_mask &= df["cell.area.px"] > 0
        df = df.loc[base_mask, ["nucleus.x.px", "nucleus.y.px", "cell.area.px"] + class_cols].copy()

        if df.empty:
            # Return an empty mask
            label_img = np.zeros((height, width), dtype=np.uint8)
            result = {name: np.zeros((height, width), dtype=bool) for name in class_cols}
            counts = {name: 0 for name in class_cols}
            return result, label_img, counts

        # Precompute integer radii from area: r = sqrt(area / pi)
        # Make sure we have at least radius 1 for tiny cells if you want them visible; otherwise keep rounding.
        radii = np.sqrt(df["cell.area.px"].values / math.pi)
        # Round to nearest int >= 1 (so very small cells still leave a dot)
        radii = np.maximum(1, np.rint(radii)).astype(np.int32)

        xs = np.rint(df["nucleus.x.px"].values).astype(np.int32)
        ys = np.rint(df["nucleus.y.px"].values).astype(np.int32)

        # ---- Create a blank label image (0 = background) ----
        # PIL expects (width, height)
        pil_img = Image.new(mode="L", size=(width, height), color=0)
        draw = ImageDraw.Draw(pil_img)

        # ---- Determine per-row label code to draw, using your one-hot columns ----
        # For each row, find which class is 1. If multiple 1s exist, we will rely on draw priority order.
        # But to stay consistent with your original code (class chosen by order of elif),
        # we will draw by class groups in the specified label_priority order.
        name_to_code = {v: k for k, v in label_names.items()}

        # Draw class-by-class to control overwrite priority
        for class_name in label_priority:
            code = int(name_to_code[class_name])  # 1..255
            # Rows where this class is active
            active = df[class_name].values.astype(bool)
            if not np.any(active):
                continue

            x_c = xs[active]
            y_c = ys[active]
            r_c = radii[active]

            # Draw each ellipse (filled circle) onto the label image with the class code.
            # Note: PIL’s ellipse draws inclusive within the bounding box.
            # Clip bounding boxes to image extent to avoid unnecessary work.
            for x, y, r in zip(x_c, y_c, r_c):
                if r <= 0:
                    continue
                left   = max(0, x - r)
                top    = max(0, y - r)
                right  = min(width - 1, x + r)
                bottom = min(height - 1, y + r)
                if right <= left or bottom <= top:
                    continue
                # Draw filled circle
                draw.ellipse([left, top, right, bottom], fill=code)

        # ---- Convert to NumPy array ----
        label_img = np.array(pil_img, dtype=np.uint8)  # (H, W)

        # ---- Per-class boolean masks ----
        result = {}
        for code, name in label_names.items():
            result[name] = (label_img == code)

        # ---- Counts ----
        binc = np.bincount(label_img.ravel(), minlength=max(label_names.keys()) + 1)
        counts = {name: int(binc[code]) for code, name in label_names.items()}
        counts_bg = int(binc[0])  # background pixels if you need it

        # ---- Optional save ----
        if out_mask_path:
            # Save as 8-bit PNG with integer classes
            Image.fromarray(label_img, mode="L").save(out_mask_path)

        # ---- Optional preview ----
        if show_preview and px is not None:
            # Map codes to readable text for hover (optional). For speed, just show label_img.
            px.imshow(label_img, title="Label mask (uint8)").show()

        # ---- Print a simple summary ----
        # (Avoiding NaN counts; boolean masks cannot contain NaN)
        total_true = sum(counts.values())
        total_pixels = int(label_img.size)
        total_false = total_pixels - total_true
        print("Per-class pixel counts:")
        for name in label_priority:
            print(f"  {name}: {counts.get(name, 0):,}")
        print(f"Background: {counts_bg:,}")
        print(f"TOTAL: True (any class)={total_true:,}, False (background)={total_false:,}, Image shape={label_img.shape}")
        
        
    
        import matplotlib.pyplot as plt

        for name in label_names.values():
            mask_img = (result[name].astype(np.uint8)) * 255
            plt.figure(figsize=(6, 6))
            plt.imshow(mask_img, cmap="gray")
            plt.title(f"Mask for class '{name}'")
            plt.axis("off")
            plt.show()

       
        

        #name_to_code = {v: k for k, v in label_names.items()}
        #result = {name_to_code[name]: mask for name, mask in result.items()}


        #rename_map = {'B': '1', 'BnT': '2','DC': '3','MacCD163': '4','Neutrophil': '5','T': '6','Tumor': '7','Tumor_CD15': '8','other': '9',}  # old_key -> new_key

        #result = {rename_map.get(k, k): v for k, v in result.items()}
        print("result keys after rename",result.keys())
        #print ("Finished Calculate_mask_cell_qptiff",result.values())
        # Count True
        for name, arr in result.items():
            num_true = np.sum(arr)  # True counts as 1
            num_false = arr.size - num_true
            print(f"{name}: True={num_true}, False={num_false}")
        # --- Convert result keys from names to integer IDs ---
        #name_to_index = {v: k for k, v in label_names.items()}
        #result = {name_to_index[name]: mask for name, mask in result.items()}
        print("result keys after rename",result.keys())        
        for name in label_names.values():
            print(name, result[name].shape, result[name].dtype, result[name].min(), result[name].max())

        print(f"lenght of the result{len(result)}, lenght array",len(result))
        #self.masks_cells_w_borders = result
        print("Canvas size (W,H):", width, height)
        print("x range:", int(x_c.min()), int(x_c.max()))
        print("y range:", int(y_c.min()), int(y_c.max()))






        # #Added new one , filtered to the ROI only
        # # Convert tile indices (1-based typical) to pixel box (x:cols, y:rows)
        # x0 = (self.origin_col - 1) * self.dataset_config.tile_width
        # y0 = (self.origin_row - 1) * self.dataset_config.tile_height
        # x1 = self.end_col * self.dataset_config.tile_width
        # y1 = self.end_row * self.dataset_config.tile_height


        x_min, x_max, y_min, y_max,_,_,_,_ = self._get_roi_w_borders_lims()

        x0 = x_min
        x1 = x_max
        y0 = y_min
        y1 = y_max

        
        # Crop (rows then cols)
        for idx, mask in result.items():
            result[idx] = mask[y0:y1, x0:x1]
            print(f"ROI mask for class '{idx}' shape:", result[idx].shape)
        
        
        for name in label_names.values():
            mask_img = (result[name].astype(np.uint8)) * 255
            plt.figure(figsize=(6, 6))
            plt.imshow(mask_img, cmap="gray")
            plt.title(f"Mask for class '{name}'")
            plt.axis("off")
            plt.show()


        self.masks_cells_w_borders = result
        return result




#####################################""
    

    @trace
    def _get_table_cells_fluorescence(self,channels_of_interest = "all", filter_cells_from_several_channels = True):
        """
        Get the table of the cells for fluorescence images

        Test if done on the entire image and do it on this particular ROI if not
        """
        # #Test if done for entire slide 
        # path_classified_slide = os.path.join(self.dataset_config.dir_classified_img, "slide_{}_cells.csv".format(self.slide_num))
        particular_roi = (self.origin_row, self.origin_col, self.end_row, self.end_col)
        path_classified_slide = classification.get_path_table_cells(self.slide_num,self.dataset_config,particular_roi = None)

        path_classified_roi = classification.get_path_table_cells(self.slide_num,self.dataset_config,particular_roi = particular_roi)
     


        # custom_csv = self.dataset_config.qptiff_tsv_path
        
        custom_csv = "/root/cloud-data/eu-ngs/U1086086/Ali_A/CSV_concate/Full_Csv1.csv"
        table_cells = pd.read_csv(custom_csv, sep=";")

        x_min, x_max, y_min, y_max, xmin_borders, xmax_borders, ymin_borders, ymax_borders = self._get_roi_w_borders_lims()
        # print("x_min, x_max, y_min, y_max, xmin_borders, xmax_borders, ymin_borders, ymax_borders",x_min, x_max, y_min, y_max, xmin_borders, xmax_borders, ymin_borders, ymax_borders)



        table_cells_in_roi = table_cells[(table_cells['x_img'] >= x_min) & (table_cells['x_img'] < x_max) & (table_cells['y_img'] >= y_min) & (table_cells['y_img'] < y_max)]
        # print("Table cell post filtering by xlim and y lim : ",table_cells_in_roi.shape)
        table_cells_in_roi['within_roi'] = True
        table_cells_in_border = table_cells[(table_cells['x_img'] >= xmin_borders) & (table_cells['x_img'] < xmax_borders) & (table_cells['y_img'] >= ymin_borders) & (table_cells['y_img'] < ymax_borders)]     
        # print("table_cells_in_border.shape after 1st filter",table_cells_in_border.shape)
        table_cells_in_border = table_cells_in_border[(table_cells_in_border['x_img'] < x_min) | (table_cells_in_border['x_img'] > x_max) | (table_cells_in_border['y_img'] < y_min) | (table_cells_in_border['y_img'] > y_max)]     
        # print("table_cells_in_border.shape after 2nd filter",table_cells_in_border.shape)

        table_cells_in_border['within_roi'] = False
        table_cells_w_borders = pd.concat([table_cells_in_roi,table_cells_in_border])
        if filter_cells_from_several_channels : 
            for cell_name in list(self.dataset_config.cells_from_multiple_channels.keys()):
                table_cells_w_borders = table_cells_w_borders[table_cells_w_borders["used_to_build_{}".format(cell_name)] == False]
        table_cells_w_borders = table_cells_w_borders.reset_index(drop=True)
        table_cells_w_borders["x_roi_w_borders"] = (table_cells_w_borders["tile_col"]-(self.origin_col))*self.dataset_config.tile_height + table_cells_w_borders["x_tile"]+self.dataset_config.roi_border_size
        table_cells_w_borders["y_roi_w_borders"] = (table_cells_w_borders["tile_row"]-(self.origin_row))*self.dataset_config.tile_width + table_cells_w_borders["y_tile"]+self.dataset_config.roi_border_size
        # x_roi_w_borders, y_roi_w_borders
        
        return  table_cells_w_borders


    @trace        
    def _get_cell_masks_fluorescence(self, channels_of_interest=None):
        # qptiff_path = self.qptiff_codes.qptiff_codes.qptiff_path
        qptiff_path = "/root/cloud-data/eu-ngs/U1086086/Ali_A/DeepCellMap/data/Immucan_F/IMMU-BC1-1140-FIXT-01-IF1-01_#_0296bfe21068affb025b0bea8706cc98.qptiff"
        info_csv = "/root/cloud-data/eu-ngs/U1086086/Ali_A/CSV_concate/Full_Csv1.csv"
        
        # Must return the dict keyed by class names ("B","T",...)


        return self.Calculate_mask_cell_qptiff(qptiff_path, info_csv)

    # def _get_cell_masks_fluorescence_including_superposition(self,channels_of_interest):
    #     """
    #     Get the mask of the cells for fluorescence images

    #     Output : 
    #     mask_cells_w_borders_by_channel : dict, keys : channel_number, values : mask of the cells for this channel
    #     """
    #     if not hasattr(self, "table_cells_w_borders"): 
    #         self.get_table_cells()

    #     mask_cells_w_borders_by_channel = {} 
    #     for channel_number in channels_of_interest:
    #         mask_cells_w_borders_by_channel[self.dataset_config.cell_class_names[channel_number-1]] = np.zeros(self.roi_w_borders_shape).astype(bool)
    #     for index, row_df in tqdm(self.table_cells_w_borders.iterrows()):
    #         channel_number, row, col, x_tile, y_tile , length_max = row_df["channel_number"], row_df["tile_row"], row_df["tile_col"], row_df["x_tile"], row_df["y_tile"], row_df["length_max"]

    #         path_cell = classification.get_path_cell(self.slide_num,channel_number, row, col, x_tile, y_tile, self.dataset_config)
    #         mask_cell = plt.imread(path_cell).astype(bool)
    #         mask_cells_w_borders_by_channel[self.dataset_config.cell_class_names[channel_number-1]] = self._add_mask_cell_in_roi_mask(mask_cells_w_borders_by_channel[self.dataset_config.cell_class_names[channel_number-1]], mask_cell,row, col, x_tile, y_tile, length_max)

    #     return mask_cells_w_borders_by_channel

    @trace
    def get_mask_cell_type(self, cell_type):
        """ Return mask(bool) with borders of type_A cells"""
        if self.dataset_config.data_type == "fluorescence":
            return self.masks_cells_w_borders[cell_type]
        elif self.dataset_config.data_type == "wsi": 
            cell_type = self.dataset_config.cell_class_names.index(cell_type)+1
            return (self.masks_cells_w_borders==cell_type)
        else : 
            raise Exception("There is no code for this kind of data")

        # #Test if done for entire slide 

        # If not, compute it on ROI
        masks_cells, masks_cells_w_borders = None, None

        return masks_cells, masks_cells_w_borders
    @trace    
    def _get_cell_masks_wsi(self):
            """
            Get the mask of the cells for WSI images

            Test if done on the entire image and do it on this particular ROI if not
            """
            if not hasattr(self, "table_cells_w_borders"): 
                self.get_table_cells()
            mask_cells_w_borders = np.zeros(self.roi_w_borders_shape, dtype = np.uint8)
            # self.table_cells_w_borders.iterrows()
            for index_cell in tqdm(range(self.table_cells_w_borders.shape[0])):
                # print("here -> aswsi _get_cell_masks_wsi")
                row_df = self.table_cells_w_borders.iloc[index_cell]
                row, col, x_tile, y_tile , length_max, cell_type = row_df["tile_row"], row_df["tile_col"], row_df["x_tile"], row_df["y_tile"], row_df["length_max"],row_df["cell_type"]
                path_cell = classification.get_path_cell(self.slide_num,0, row, col, x_tile, y_tile, self.dataset_config)
                # print("path_cell : ",path_cell)
                mask_cell = plt.imread(path_cell)
                # if x_tile == 618 and y_tile == 186 :
                #     print("row, col, x_tile, y_tile , length_max, decision",row, col, x_tile, y_tile , length_max, decision)
                #     print("np.unqique(mask_cell)", np.unique(mask_cell))
                #     display_mask(mask_cell,title="mask_cell[_get_cell_masks_wsi]")
                mask_cells_w_borders = self._add_mask_cell_in_roi_mask(mask_cells_w_borders, mask_cell,row, col, x_tile, y_tile, length_max, cell_type = cell_type)
            return mask_cells_w_borders

    @trace
    def save_mask_cells(self):
        path_mask = os.path.join(self.path_roi, "masks_cells_w_borders.png")
        # np.save(path_mask, roi.masks_cells_w_borders)
        print(path_mask)
        mask_img = np_to_pil(self.masks_cells_w_borders)
        mask_img.save(path_mask)
    @trace
    def load_mask_cells(self):
        path_mask = os.path.join(self.path_roi, "masks_cells_w_borders.png")
        mask_img = np.asarray(Image.open(path_mask))
        return mask_img
    @trace
    def get_nuclei_mask(self): 
        from cellpose import models
        tile_width = self.dataset_config.tile_width
        tile_height = self.dataset_config.tile_height
        roi_border_size = self.dataset_config.roi_border_size
        param_cellpose = self.dataset_config.param_best_cellpose
        model_type = param_cellpose["model_type"]
        diameter = param_cellpose["diameter"]
        channels = param_cellpose["channels"]
        normalisation = param_cellpose["normalisation"]
        net_avg = param_cellpose["net_avg"]
        model = models.Cellpose(model_type=model_type)
        if self.dataset_config.data_type == "fluorescence":
            rgb = self.image_w_borders["RGB"]
        elif self.dataset_config.data_type == "wsi":
            rgb = self.image_w_borders
        else :
            raise Exception("There is no code for this kind of data")
        
        n_tiles_row = (self.end_row-self.origin_row+1)
        n_tiles_col = (self.end_col-self.origin_col+1)
        mask_width = (n_tiles_col+2)*tile_width
        mask_heigh = (n_tiles_row+2)*tile_height
        mask_nuclei = np.zeros((mask_heigh,mask_width))

        rgb_padded = np.pad(rgb, ((tile_width - roi_border_size,tile_width -roi_border_size),(tile_width-roi_border_size,tile_width-roi_border_size),(0,0)),"constant" ) #0 padding

        for row in range(n_tiles_row+2):
            for col in range(n_tiles_col+2):

                rgb_tile = rgb_padded[row*tile_width:(row+1)*tile_width,col*tile_height:(col+1)*tile_height,:]
                mask_tile, flows, styles, diam = model.eval([rgb_tile], diameter=diameter, channels=channels,normalize=normalisation, net_avg = net_avg)
                mask_tile = mask_tile[0]
                mask_nuclei[row*tile_width:(row+1)*tile_width,col*tile_height:(col+1)*tile_height] = mask_tile
        
        
        mask = mask_nuclei[(tile_width-roi_border_size):-(tile_width-roi_border_size),(tile_width-roi_border_size):-(tile_width-roi_border_size)]

        self.mask_nuclei = mask
    @trace
    def display_nuclei(self,with_other_cells = True, save_fig = True,color_name="dataset_config",output_path_name="",roi_category="", with_background = True, with_roi_delimiter=False,with_anatomical_part_mask=False,figsize=(20,20)):
        to_expo = True #Bc this github branch of the project 
        with_tiles_delimitation = False
        with_center_of_mass=False
        if not hasattr(self,"image_w_borders"): 
            with_background = False
        if with_background : 
            if self.dataset_config.data_type == "fluorescence":
                if self.dataset_config.consider_image_with_channels :
                    background = self.image_w_borders["RGB"]
                else :
                    # background = np.dstack([self.image_w_borders[:,:,1], self.image_w_borders[:,:,1], self.image_w_borders[:,:,1]])
                    background = self.image_w_borders

            elif self.dataset_config.data_type == "wsi": 
                background = self.image_w_borders
            else : 
                raise Exception("There is no code for this kind of data")
        else : 
            background = np.zeros((self.roi_w_borders_shape[0],self.roi_w_borders_shape[1],3))
        background_pil = np_to_pil(background)
        drawing = ImageDraw.Draw(background_pil, "RGBA")

        drawing, nb_cells = draw_nuclei_on_img(drawing,mask = self.mask_nuclei) #add_cells_to_draw
        print("drawing", type(drawing))
        if with_other_cells : 
            drawing = draw_cells_on_img(self, drawing,color_name=color_name) #add_cells_to_draw
        if with_roi_delimiter : 
            drawing = draw_roi_delimiter(self, drawing)
        if with_anatomical_part_mask:
            drawing = draw_anatomical_part_mask(self,drawing)
        if with_center_of_mass : 
            print("with_center_of_mass TODO ")
        if with_tiles_delimitation:
            drawing = with_tiles_delimitations(self, drawing)


        fig = plt.figure(figsize=figsize)  #, tight_layout = True)
        plt.imshow(background_pil)
        plt.axis('off')
        if save_fig:
            if to_expo : 
                directory = os.path.join(self.dataset_config.dir_output,OUTPUT_EXPO_NAME,output_path_name,roi_category)
                os.makedirs(directory, exist_ok=True)
                figname = "s"+str(self.slide_num).zfill(3)+"_ro" +str(self.origin_row).zfill(3) + "_co" +str(self.origin_col).zfill(3) + "_re" +str(self.end_row).zfill(3) + "_ce" +str(self.end_col).zfill(3)

                path_save = find_path_last(directory,figname+"_nuclei" )
                print("path_save",path_save)
                fig.savefig(path_save,
                            facecolor='white',
                            dpi="figure",
                            bbox_inches='tight',
                            pad_inches=0.1)
            else : 

                path_save = find_path_last(self.path_roi, "_nuclei")
                print("path_save",path_save)
                fig.savefig(path_save,
                            facecolor='white',
                            dpi="figure",
                            bbox_inches='tight',
                            pad_inches=0.1)


    @trace
    def display_segmented_cells(self, channels_of_interest = None,figsize = (30,30),color_name="dataset_config",with_background =True,with_roi_delimiter = False ,with_anatomical_part_mask = False , with_center_of_mass = False,  save_fig =True,output_path_name="segmented_cells_on_tissue",roi_category="all"):
        """ channels_of_interest may be changed by :
        cells_of_interest_by_channels = ["1","2","1+2"]
        """
        to_expo = True #Bc this github branch of the project 
        with_tiles_delimitation = False
        if not hasattr(self,"image_w_borders"): 
            with_background = False
                   
        if with_background : 
            if self.dataset_config.data_type == "fluorescence":
                if self.dataset_config.consider_image_with_channels :
                    background = self.image_w_borders["RGB"]
                else :
                    # background = np.dstack([self.image_w_borders[:,:,1], self.image_w_borders[:,:,1], self.image_w_borders[:,:,1]])
                    background = self.image_w_borders

            elif self.dataset_config.data_type == "wsi": 
                background = self.image_w_borders
            else : 
                raise Exception("There is no code for this kind of data")
        else : 
            background = np.zeros((self.roi_w_borders_shape[0],self.roi_w_borders_shape[1],3))

        background_pil = np_to_pil(background)
        
        drawing = ImageDraw.Draw(background_pil, "RGBA")
        print("background_pil",type(background_pil))
        if self.dataset_config.data_type == "fluorescence":
            if channels_of_interest == None :
                print("None") 
                channels_of_interest = self.dataset_config.channels_of_interest
            print("channels_of_interest",channels_of_interest)
            drawing = draw_cells_on_img(self, drawing, cells_of_interest=channels_of_interest,color_name=color_name)
        else : #microglia
            drawing = draw_cells_on_img(self, drawing,color_name=color_name) #add_cells_to_draw
        if with_roi_delimiter : 
            drawing = draw_roi_delimiter(self, drawing)
        if with_anatomical_part_mask:
            drawing = draw_anatomical_part_mask(self,drawing)
        if with_center_of_mass : 
            drawing = draw_center_of_mass(self,drawing,channels_of_interest)
        if with_tiles_delimitation:
            drawing = with_tiles_delimitations(self, drawing)
        fig = plt.figure(figsize=figsize)  #, tight_layout = True)
        plt.imshow(background_pil)
        plt.axis('off')
        if save_fig:
            if to_expo : 
                directory = os.path.join(self.dataset_config.dir_output,OUTPUT_EXPO_NAME,output_path_name,roi_category)
                os.makedirs(directory, exist_ok=True)
                figname = "s"+str(self.slide_num).zfill(3)+"_ro" +str(self.origin_row).zfill(3) + "_co" +str(self.origin_col).zfill(3) + "_re" +str(self.end_row).zfill(3) + "_ce" +str(self.end_col).zfill(3)

                path_save = find_path_last(directory,figname+"_segmented_cells_on_tissue" )

                fig.savefig(path_save,
                            facecolor='white',
                            dpi="figure",
                            bbox_inches='tight',
                            pad_inches=0.1)
            else : 

                path_save = find_path_last(self.path_roi, "segmented_cells_on_tissue")
                fig.savefig(path_save,
                            facecolor='white',
                            dpi="figure",
                            bbox_inches='tight',
                            pad_inches=0.1)
    @trace
    def _get_cells_statistics(self, dict_statistics_roi):
        
        image_statistic = pd.read_csv(os.path.join(self.dataset_config.dir_output_dataset, "statistics_images.csv"), sep = ";")
        image_statistic_dict = image_statistic[image_statistic["slide_num"] == self.slide_num].iloc[0].to_dict()
        table_cells_roi = self.table_cells_w_borders[self.table_cells_w_borders["within_roi"] == True]
        #if not hasattr(self, "table_cells_roi"):
        #    self.table_cells_roi = self.table_cells_w_borders[self.table_cells_w_borders["within_roi"] == True].copy()

        
        dict_statistics_roi["n_cells_roi"] = table_cells_roi.shape[0] 
        dict_statistics_roi["n_cells_roi_w_border"] = self.table_cells_w_borders.shape[0]
        dict_statistics_roi["fraction_tot_cells_in_roi"] = dict_statistics_roi["n_cells_roi"]/image_statistic_dict["n_cells_slide"]
        dict_statistics_roi["mean_n_cells_per_tile_roi"] = table_cells_roi.groupby(["tile_row","tile_col"])['id_cell'].count().mean()
        dict_statistics_roi["std_n_cells_per_tile_roi"] = table_cells_roi.groupby(["tile_row","tile_col"])['id_cell'].count().std()
        dict_statistics_roi["mean_cell_size_roi"] = table_cells_roi['size'].mean()
        dict_statistics_roi["std_cell_size_roi"] = table_cells_roi['size'].std()
        
        for idx_decision, cell_type_name in enumerate(self.dataset_config.cell_class_names,1):
            dict_statistics_roi["n_cells_{}_roi".format(cell_type_name)] = table_cells_roi[table_cells_roi["cell_type"] == idx_decision].shape[0]
            dict_statistics_roi["n_cells_{}_roi_w_border".format(cell_type_name)] = self.table_cells_w_borders[self.table_cells_w_borders["cell_type"] == idx_decision].shape[0]
            dict_statistics_roi["n_all_cells_except_{}_roi".format(cell_type_name)] = table_cells_roi[table_cells_roi["cell_type"] != idx_decision].shape[0]
            dict_statistics_roi["fraction_{}_roi".format(cell_type_name)] = table_cells_roi[table_cells_roi["cell_type"] == idx_decision].shape[0]/dict_statistics_roi["n_cells_roi"] if dict_statistics_roi["n_cells_roi"]!= 0 else None 
            dict_statistics_roi["mean_size_{}_roi".format(cell_type_name)] = table_cells_roi[table_cells_roi['cell_type'] == idx_decision]['size'].mean()
            dict_statistics_roi["std_size_{}_roi".format(cell_type_name)] = table_cells_roi[table_cells_roi['cell_type'] == idx_decision]['size'].std()
            if image_statistic_dict["n_cells_{}_slide".format(cell_type_name)] != 0 : 
                dict_statistics_roi["fraction_total_{}_in_roi".format(cell_type_name)] = table_cells_roi[table_cells_roi["cell_type"] == idx_decision].shape[0]/image_statistic_dict["n_cells_{}_slide".format(cell_type_name)]
            else : 
                dict_statistics_roi["fraction_total_{}_in_roi".format(cell_type_name)] = np.nan

            if self.dataset_config.statistics_with_proba:
                dict_statistics_roi["n_cells_{}_proba_roi".format(cell_type_name)] = table_cells_roi["proba_{}".format(cell_type_name)].sum()
                dict_statistics_roi["n_cells_{}_proba_roi_w_border".format(cell_type_name)] = self.table_cells_w_borders["proba_{}".format(cell_type_name)].sum()
                dict_statistics_roi["fraction_{}_proba_roi".format(cell_type_name)] = table_cells_roi["proba_{}".format(cell_type_name)].sum()/dict_statistics_roi["n_cells_roi"]
                if image_statistic_dict["n_cells_{}_proba_slide".format(cell_type_name)] != 0 :
                    dict_statistics_roi["fraction_total_{}_proba_in_roi".format(cell_type_name)] = dict_statistics_roi["n_cells_{}_proba_roi".format(cell_type_name)]/image_statistic_dict["n_cells_{}_proba_slide".format(cell_type_name)]
                else : 
                    dict_statistics_roi["fraction_total_{}_proba_in_roi".format(cell_type_name)] = np.nan

        return dict_statistics_roi
    @trace    
    def _get_tissue_statistics_roi(self, dict_statistics_roi):

        mask_tissue = self.mask_tissue_w_borders[self.dataset_config.roi_border_size:-self.dataset_config.roi_border_size,self.dataset_config.roi_border_size:-self.dataset_config.roi_border_size]
        dict_statistics_roi["area_tissue_roi"] = np.sum(mask_tissue)
        dict_statistics_roi["area_physiological_part_roi"] = None
        dict_statistics_roi["fraction_tissue_roi"] = dict_statistics_roi["area_tissue_roi"]/dict_statistics_roi["area_roi"]
        dict_statistics_roi["fraction_physiological_part_roi"] = None
        dict_statistics_roi["fraction_tot_tissue_in_roi"] = dict_statistics_roi["area_tissue_roi"]/dict_statistics_roi["area_tissue_slide"]
        dict_statistics_roi["fraction_tot_physiological_part_in_roi"] = None
        return dict_statistics_roi
    @trace    
    def _get_nuclei_statistics(self, dict_statistics_roi):
        dict_statistics_roi["n_nuclei_in_roi"] = None
        dict_statistics_roi["mean_nuclei_density_roi"] = None
        dict_statistics_roi["std_nuclei_density_roi"] = None
        dict_statistics_roi["ratio_nuclei_density_roi_vs_slide"] = None
        return dict_statistics_roi
    @trace   
    def _get_dataset_specific_statistics_roi(self,dict_statistics_roi):
        #parameters
        for feature_name in self.dataset_config.colnames_df_roi:
            if feature_name in dict_statistics_roi.keys():
                continue 
            else :
                if feature_name in "exemple":
                    dict_statistics_roi[feature_name] = None
                else : 
                    print("feature_name",feature_name)
                    raise Exception("Feature name not found")

        return dict_statistics_roi
    @trace
    def get_statistics_roi(self, channels_of_interest = None, save = False):
        """
        Get the statistics of the ROI
        Test if done on the entire image and do it on this particular ROI if not
        """
        statistic_image = pd.read_csv(os.path.join(self.dataset_config.dir_output_dataset, "statistics_images.csv"), sep = ";")
        print("statistic_image head:",statistic_image.head())
        dict_statistics_roi = statistic_image[statistic_image["slide_num"] == self.slide_num].iloc[0].to_dict()

        dict_statistics_roi["roi_loc"] = (self.origin_row,self.origin_col,self.end_row,self.end_col)
        dict_statistics_roi["origin_row"] = self.origin_row
        dict_statistics_roi["origin_col"] = self.origin_col
        dict_statistics_roi["end_row"] = self.end_row
        dict_statistics_roi["end_col"] = self.end_col
        dict_statistics_roi["n_tiles_row_roi"] = dict_statistics_roi["end_row"] - dict_statistics_roi["origin_row"] +1
        dict_statistics_roi["n_tiles_col_roi"] = dict_statistics_roi["end_col"] - dict_statistics_roi["origin_col"] +1
        dict_statistics_roi["roi_shape"] = self.roi_shape
        dict_statistics_roi["roi_shape_in_tiles"] = (dict_statistics_roi["n_tiles_row_roi"],dict_statistics_roi["n_tiles_col_roi"])
        dict_statistics_roi["roi_height"] = dict_statistics_roi["n_tiles_row_roi"]*self.dataset_config.tile_height
        dict_statistics_roi["roi_width"] = dict_statistics_roi["n_tiles_col_roi"]*self.dataset_config.tile_width
        dict_statistics_roi["roi_border_size"] = self.dataset_config.roi_border_size
        dict_statistics_roi["area_roi"] = dict_statistics_roi["roi_height"]*dict_statistics_roi["roi_width"]
        print("dict_statistics_roi before cells",dict_statistics_roi)
        
        dict_statistics_roi = self._get_cells_statistics(dict_statistics_roi)
        dict_statistics_roi = self._get_tissue_statistics_roi(dict_statistics_roi)
        dict_statistics_roi = self._get_nuclei_statistics(dict_statistics_roi)
        #dict_statistics_roi = self._get_dataset_specific_statistics_roi(dict_statistics_roi)
        
        print("dict_statistics_roi after cells",dict_statistics_roi)
        dict_statistics_roi_df = pd.DataFrame([dict_statistics_roi])
        print("dict_statistics_roi_df",dict_statistics_roi_df)
        if save :
            path_statistics_roi = os.path.join(self.path_roi,"statistics_roi.csv")
            dict_statistics_roi_df.to_csv(path_statistics_roi, sep = ";", index = False)
        
        self.dict_statistics_roi_df = dict_statistics_roi_df
        print("dict_statistics_roi_df",self.dict_statistics_roi_df)
    @trace
    def get_df_entire_image(self):
        """Once 3 statistical modules have been computed, get the dataframe of the entire image
        """

        from code.stat_analysis.colocalisation_analysis import ColocAnalysis
        from code.stat_analysis.dbscan_analysis import DbscanAnalysis
        from code.stat_analysis.neighbours_analysis import NeighborsAnalysis

        path_statistics_roi = os.path.join(self.path_roi,"statistics_roi.csv")
        path_colocalisation_roi = os.path.join(self.path_roi,"2_cell_cell_colocalisation")
        path_dbscan_roi = os.path.join(self.path_roi,"3_DBSCAN")
        path_neighbors_analysis_roi = os.path.join(self.path_roi,"4_neighbors_analysis")

        if os.path.exists(path_statistics_roi):
            df_roi = pd.read_csv(path_statistics_roi, sep = ";")
        else :
            raise Exception("No statistics_roi.csv found in {}".format(self.path_roi))
        
        if os.path.exists(path_colocalisation_roi):
            df_coloc = pd.DataFrame(columns = ColocAnalysis.colnames_colocalisation + ["coloc_parameter_set","levelsets"])
            for idx_parameter_set, parameter_set in enumerate([k for k in os.listdir(path_colocalisation_roi) if "DS" not in k]):
                path_folder_coloc = os.path.join(path_colocalisation_roi,parameter_set)
                df = pd.read_csv(os.path.join(path_folder_coloc,"colocalisation.csv"), sep = ";")
                df["coloc_parameter_set"] = idx_parameter_set
                df["levelsets"] = parameter_set
                df_coloc = pd.concat([df_coloc,df], axis = 0)
            df_coloc = df_coloc.reset_index(drop=True)

            # print("df_coloc.shape",df_coloc.shape)

        if os.path.exists(path_dbscan_roi): 
            df_dbscan = pd.DataFrame(columns = DbscanAnalysis.colnames_dbscan_statistics)
            for idx_parameter_set, parameter_set in enumerate([k for k in os.listdir(path_dbscan_roi) if "DS" not in k]):
                path_folder = os.path.join(path_dbscan_roi,parameter_set)
                df = pd.read_csv(os.path.join(path_folder,"dbscan_statistics.csv"), sep = ";")
                df_dbscan = pd.concat([df_dbscan,df], axis = 0)
            df_dbscan = df_dbscan.reset_index(drop=True)
            # print("df_dbscan.shape",df_dbscan.shape)

        
        if os.path.exists(path_neighbors_analysis_roi):
            path_df_neighbors_naalysis = os.path.join(path_neighbors_analysis_roi, "df_neighbours_analysis.csv")
            df_neighbors_analysis = pd.read_csv(path_df_neighbors_naalysis, sep = ";")
            # print("df_neighbors_analysis.shape",df_neighbors_analysis.shape)

        df_statistics = df_coloc.merge(df_dbscan, on = ["type_A","type_B"],how='outer')
        # print("coloc_dvbscan merge.shape",df_statistics.shape)
        # print("df_statistics.columns ",df_statistics.columns) 

        df_statistics = df_statistics.merge(df_neighbors_analysis, on = ["type_A","type_B"],how='outer')
        # print("coloc_dvbsca_nnn merge.shape",df_statistics.shape)

        # Replicate values of the first DataFrame along all rows of the final DataFrame
        replicated_df_roi = pd.concat([df_roi] * len(df_statistics), ignore_index=True, axis=0)
        results_roi = pd.concat([replicated_df_roi,df_statistics], axis=1)
        results_roi.to_csv(os.path.join(self.path_roi,"results_roi.csv"), sep = ";", index = False) 
        self.results_roi = results_roi

    #Extract some stats from roi 
    @trace
    def get_cell_number_from_type(self,cell_type, with_proba=False):
        if with_proba:
            return self.dict_statistics_roi_df.iloc[0].to_dict()["n_cells_{}_proba_roi".format(cell_type)]
        else :
            print(self.dict_statistics_roi_df.iloc[0].to_dict()["n_cells_{}_roi".format(cell_type)])
            return self.dict_statistics_roi_df.iloc[0].to_dict()["n_cells_{}_roi".format(cell_type)]
              
    @trace
    def get_feature_from_statistic(self,feature_name):
        return self.dict_statistics_roi_df.iloc[0].to_dict()[feature_name]
    @trace
    def at_least_one_B_cell(self, cell_types_B):
        """ If no B at all, no need to compute distance map 
        """
        for type_B in cell_types_B:
            if self.get_cell_number_from_type(type_B) > 0:
                return True
        return False
