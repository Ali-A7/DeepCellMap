# config/datase_immucan_1140_config.py
from types import SimpleNamespace
# config/datase_immucan_1140_config.py
from types import SimpleNamespace
import os
import os
import json
from config.base_config import BaseConfig
from config.datasets_config import DatasetBaseConfig, PreprocessingConfig
class ImmuCan1140Config:
    """
    Standalone config for ImmuCan IF1 / QPTIFF tiling + TSV integration.
    Adjust qptiff_channel_map if your channel order differs.
    """

    def __init__(self):
        
        
        
        self.processing_mode = "single"         # "single" OR "all"
        self.cohort_dir = "/root/cloud-data/cloud-pipeline-immucan-storage/IF/BC1/IF1"              # path to folder containing qptiff/ and tsv/
        self.single_image_name = "IMMU-BC1-1140-FIXT-01-IF1-01_#_0296bfe21068affb025b0bea8706cc98.qptiff"        # ONLY used in single mode (full filename)
        self.output_base_dir = "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/ROIs_filtered/1140/all_rois"   
        
        self.mapping_img_name = ["001"]
        self.dataset_name = "NewDataset3"

        self.dir_base_roi = "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/tiles_filtered/1140/masks"
        self.dir_classified_img = "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/classified_images"
        self.dir_dataset = "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/data/NewDataset3"
        self.data_format = "qptiff"  # or "tiff"/"tif" if your file ends with that
        self.dir_output_dataset = "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3"
        # Map slide stems to absolute file paths (used here because you gave exact paths)
        # Keys should match slide file stems (e.g., '001')
        self.custom_slide_paths = {
            "001": "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/data/NewDataset3/001.qptiff"
        }
         
        # Map slide stems to TSV cell table paths (optional but provided)
        self.custom_cells_table_paths = {
            # You gave '000.tsv.gz' for the first slide; we map it to '001' to pair with 001.qptiff
            "001": "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/data/immucan/tables/000.tsv.gz"
        }
        
        # self.qptiff_dir = "/root/cloud-data/cloud-pipeline-immucan-storage/IF/BC1/IF1/qptiff"
        # self.config.tsv_dir = "/root/cloud-data/cloud-pipeline-immucan-storage/IF/BC1/IF1//tsv"
        self.qptiff_tsv_path = "/root/cloud-data/eu-ngs/U1086086/Ali_A/CSV_concate/Full_Csv1.csv"


        # ---- Channels & mask ----
        # Default: first 8 pages are DAPI, CD15, CK, CD3, CD11c, CD20, CD163, AF
        # If preview looks wrong, update indices below to the right pages
        self.qptiff_channel_map = {
            "DAPI":  0,
            "CD15":  1,
            "CK":    2,
            "CD3":   3,
            "CD11c": 4,
            "CD20":  5,
            "CD163": 6,
            "AF":    7,
        }
        self.dapi_key = "DAPI"

        # For RGB previews (summaries/top tiles)
        self.preview_rgb = {
            "R": ["CK", "CD163", "CD20", "CD11c", "CD15"],
            "G": ["CD3", "CD11c", "CD20", "CD15"],
            "B": ["DAPI"],
        }
        
 
        # legacy-compatible alias used by ROI code: dataset_config.preprocessing_config.scale_factor  # [1](https://sanofi-my.sharepoint.com/personal/ali_ahmadi_sanofi_com/Documents/Fichiers%20Microsoft%20Copilot%20Chat/datase_immucan_1140_config.py)
        self.scale_factor = 8
        #self.preprocessing_config = SimpleNamespace(scale_factor=self.scale_factor)

  # NEW: mask parameters for both preview and tiling 
        self.mask_params = SimpleNamespace(
            mode="otsu_loose",   # 'otsu' | 'otsu_loose' | 'percentile'
            alpha=0.85,          # lower than 1.0 => more inclusive than Otsu
            percentile=None,     # e.g., 70.0 if you prefer percentile mode
            invert=False,        # True only if nuclei are darker than background
            sigma=1.2,           # slightly lower blur keeps fine detail
            min_object_px=50,    # smaller fragments kept
            hole_size_px=100,    # fill smaller holes
            dilate_px=1          # tiny dilation to connect fragments
        )



        # ---- Tiling ----
        self.tile_size = 1024       # you asked 1024 in your earlier config
        self.stride = 1024          # set < tile_size (e.g. 512) for overlap
        self.min_foreground_frac = 0.25
        self.top_k = 64
        self.save_tiles_as = "png"  # or "npy"

        # ---- Misc ----
        self.verbose = True
        # Pixel size, if known (µm/px), not used here but useful downstream
        self.conversion_px_micro_meter = 0.5

       
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



        #self.dir_output_dataset = "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3" 
        
        self.dir_output = "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3" 

        # ---- Channels (keep as-is; adjust if needed) ----
        self.qptiff_channel_map = {
            "DAPI":  0, "CD15": 1, "CK": 2, "CD3": 3,
            "CD11c": 4, "CD20": 5, "CD163": 6, "AF": 7,
        }
        self.dapi_key = "DAPI"

        # ---- Preview/downscale factor (keep simple) ----
        self.scale_factor = 8
        #self.preprocessing_config = SimpleNamespace(scale_factor=self.scale_factor)

        # ---- Tiling params (used by Notebook 1 and the CSV) ----
        self.tile_size = 1024
        self.stride = 1024  # = tile_size (no overlap)

        # ---- Optional thresholds for selecting “informative” tiles ----
        self.min_foreground_frac = 0.25
        self.top_k = 64
        self.save_tiles_as = "png"

        # ---- Pixel size fallback (µm/px) ----
        self.conversion_px_micro_meter = 0.5

       
       
       
       
       
       
       # Orginal _________________________________________________
 

        self.mapping_img_gender = None
        self.mapping_img_disease = []
        self.mapping_img_age = []
        self.cell_class_names = ["B", "BnT", "DC", "MacCD163", "Neutrophil", "T", "Tumor", "Tumor_CD15", "other"]
        self.debug_mode = False
        self.tile_border_size = 0
        self.data_type = "fluorescence"
        self.consider_image_with_channels = True 
        #self.data_format = "czi"
        self.conversion_px_micro_meter = 1 
 

        # C1:DAPI-T1(blue) , C2: AF568-T2(red) , C3:AF647-T3(white), C4(AF488-T4(Green) )
        self.channel_names = {            
            "B": 1,
            "BnT":  2,
            "DC":  3,
            "MacCD163":   4,
            "Neutrophil": 5,
            "T":  6,
            "Tumor": 7,
            "Tumor_CD15": 8,
            "other": 9,
            }
        self.dim_position = dict({"C":0,"X":2,"Y":3,"Z":1})
        self.has_Z = True
        #Tiling parameters
        self.tile_width = 1024
        self.tile_height = 1024
        self.roi_border_size = int(1024/4)
        self.border_size_during_segmentation = int(1024/4)
        self.crop_size = 256
        self.scale_factor = 1
        self.tissue_extraction_accept_holes = False
        self.dataset_name = "NewDataset3"
        # Segmentation parameters
        self.preprocessing_config = PreprocessingConfig(dataset_name = self.dataset_name,
                                                        scale_factor=self.scale_factor,
                                                        tissue_extraction_accept_holes=self.tissue_extraction_accept_holes)

        self.tissue_segmentation_param = dict({
            "default" : dict({
                "manual_threshold" : 5,
                "dilation" : 25,
                "fill_holes" : None,
                "remove_small_objects" : 20000,
                "dilation2" : 20,
                "take_largest_component" : 1}),

        })
        self.threshold_tissue = 0.95
        self.save_tissue_segmentation_steps = False 
        self.channel_used_to_segment_tissue = 0

        # Classification parameters
        self.use_imgs_as_channels = False 
        self.channels_cells_to_segment = [1,2,3,4]
        self.channels_of_interest = [1,2,3,4,5,6,7,8,9]
        
        self.cell_class_names = [
                 "B", "BnT", "DC", "MacCD163",
                "Neutrophil", "T", "Tumor", "Tumor_CD15", "other"
                            ]

        self.cells_from_multiple_channels = dict({
            "exemple_of_celltype_composed_from_several_channels" : [1,2]
        })
        self.association_cell_name_channel_number = dict({
            "cell_type_1" : 1,
            "cell_type_2" : 2,
            "exemple_of_celltype_composed_from_several_channels" : 4
        })
        self.param_best_cellpose = dict({"model_type" : "cyto2","diameter" : 20,"channels" : [[0,0]], "normalisation" : True, "net_avg" : False})
        self.cellpose_parameters = dict(
            {
                "downscale_factor_cellpose_tissue_segmentation": None,  # better : scale_factor_tile_crop_cellpose
                "tile_subdivision_factor" : 16,
                "channel_nuclei" : 0
            }
        )
        self.model_segmentation_name = "image_processing_steps"
        self.cell_segmentation_param = None

        self.cell_segmentation_param_by_cannnel = dict({
            "default": dict({
                0: {
                "cellpose" : self.param_best_cellpose
                },
                1: {
                    "multi_otsu_thresholding" :  [None, 40],
                    "dilation" : 8,
                    "filter_center_cells" : self.tile_width,
                    "remove_small_objects" : 250
                    },
                2: {#Ok pour slide 002, NOT FOR 1 
                    "multi_otsu_thresholding" :  [None, 40],
                    # "otsu_thresholding" :  None,
                    "erosion": 2,
                    "dilation" : 4,
                    "filter_center_cells" : self.tile_width,
                    "fill_holes" : None,
                    "remove_small_objects" : 200,
                    "remove_large_objects" : 10000
                    },
                3: {#Ok pour slide 002, 
                    "multi_otsu_thresholding" :  [None, 40],
                    # "otsu_thresholding" :  None,
                    # "erosion": 1,
                    "dilation" : 2,
                    "filter_center_cells" : self.tile_width,
                    "fill_holes" : None,
                    "remove_small_objects" : 250,
                    "remove_large_objects" : 15000
                    }
                }),
                })

        self.classification_param = None
        self.cell_class_names_for_classification =  [   "B", "BnT", "DC", "MacCD163",
                "Neutrophil", "T", "Tumor", "Tumor_CD15", "other" ]
                            
        self.tile_test_segmentation = dict({
            "001": [(5,5),(3,4),(3,5),(2,5),(5,3),(4,2),(5,2),(4,7),(5,4)],
            "003" : [(3,10),(4,2),(4,9),(2,2),(4,3), (4,5),(5,5),(5,4),(5,3),(2,5),(4,6),(5,5),(4,4)],
            "004" : [(1,6),(4,1),(2,1),(2,3),(2,7),(1,4),(1,5),(3,2), (4,2), (4,1), (2,6)],
        })
        self.roi_test_tissue_border = [(2,2,2,3,3)]
        self.roi_cool = [(1,17,35,19,37)]
        #Usage : 
        #slide_num, origin_row, origin_col,end_row, end_col  = dataset_config.roi_test_roi_4_tiles[0]
        


        self.statistics_with_proba = False


        #self.path_cells = os.path.join(self.dir_base_classif,"cells_per_images","cells")
        self.physiological_regions_max_square_size = None
        self.physiological_regions_group_for_comparaison = None


        #Colocalisation 
        self.cell_cell_colocalisation_config = dict(
            {   "compute_with_proba" : False,
                "cell_types_A": self.cell_class_names,
                "cell_types_B": self.cell_class_names,
                "levelsets": [0,50,100,150,200,250,300,400,500,600,700,800,900,1000,1100,1200],
                "save_images": True,
            }
        )

        self.dbscan_based_analysis_config = dict(
            {
            "min_sample": 3,
            "range_epsilon_to_test" : [100,200,300,400,500,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5600,5800,6000,6300,6600,7000,7300,7600,8000,8500,9000] ,
            "cell_types_A": self.cell_class_names,
            "cell_types_B": self.cell_class_names,
            "config_cluster_robustess_experiment": {
                "n_experiment_of_removing": 100,
                "ratio_removed_cells_robustess_test": 0.1,
                "threshold_conserved_area": 0.6
            },
            "display_convex_hull_clusters": True,
            "save_figure_dbscan": True
            }
        )
        self.neighbors_analysis_config = dict(
            {
            "n_closest_neighbors_of_interest": 3,
            "cell_types_A": self.cell_class_names,
            "cell_types_B": self.cell_class_names,
            }
        )
        #methods 
        # self.create_path()
        self.colnames_table_cells_base = [
            "id_cell",
            "cell_type",
            "channel_number",
            "tile_row",
            "tile_col",
            "tile_cat",
             "x_tile", 
             "y_tile", 
             "x_img", 
             "y_img",
             "x_tile_border",
             "y_tile_border",
            "size",
            "length_max",
            "check_out_of_borders",
            "check_in_centered_tile",
            ]
        self.colnames_df_image = (
            ["slide_num",
            "slide_shape",
            "area_slide",
            "slide_shape_in_tile",
            "n_tiles_slide",
            "pixel_resolution",
            # "gender",
            
            "area_tissue_slide",
            "fraction_tissue_slide",

            ####parameters
            # "model_segmentation_slide", 
            # "model_classification_slide",

            #####results cells 
            "n_nuclei_in_slide",
            "mean_nuclei_density_slide",
            "std_nuclei_density_slide",
        
            #####all cells
            "n_cells_slide",
            "mean_n_cells_per_tile_slide",
            "std_n_cells_per_tile_slide",
            "mean_cell_size_slide",
            "std_cell_size_slide"]

            #### cell type A
            #cell type A
            + ["n_cells_{}_slide".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["fraction_{}_slide".format(cell_type_name) for cell_type_name in self.cell_class_names]

            + ["mean_size_{}_slide".format(cell_type_name) for cell_type_name in self.cell_class_names] 
            + ["std_size_{}_slide".format(cell_type_name) for cell_type_name in self.cell_class_names]  
        )
        self.colnames_df_roi = (
            [
            #parameters 
            "roi_loc",
            "origin_row",
            "origin_col",
            "end_row",
            "end_col",
            "n_tiles_row_roi",
            "n_tiles_col_roi",
            "roi_shape",
            "roi_shape_in_tiles",
            "roi_height",
            "roi_width",
            "area_roi",
            "area_tissue_roi",
            # "area_physiological_part_roi",
            "fraction_tissue_roi",
            # "fraction_physiological_part_roi",
            "fraction_tot_tissue_in_roi",
            # "fraction_tot_physiological_part_in_roi",

            #results cells 
            "n_nuclei_in_roi",
            "mean_nuclei_density_roi",
            "std_nuclei_density_roi",
            "ratio_nuclei_density_roi_vs_slide",

            #all cells 
            "n_cells_roi",
            "fraction_tot_cells_in_roi",
            "mean_n_cells_per_tile_roi",
            "std_n_cells_per_tile_roi",
            "mean_cell_size_roi",
            "std_cell_size_roi"]

            #cell type A
            + ["n_cells_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["fraction_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            # + ["n_cells_{}_proba_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            # + ["fraction_{}_proba_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["mean_size_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names] 
            + ["std_size_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]  
            #Comparaison with entire slide 
            + ["fraction_total_{}_in_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]  
            # + ["fraction_total_{}_proba_in_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]  
        )


    def save(self):
        """
        Save the configuration to a JSON file.
        """
        path_to_save = os.path.join(self.dir_config, self.dataset_name,"config.json")
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        super().save(path_to_save)


