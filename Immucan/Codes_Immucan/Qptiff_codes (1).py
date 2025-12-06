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
import os
import numpy as np
import pandas as pd
from PIL import Image
import os
import re
import pandas as pd
import gzip
import tifffile as tiff
from segmentation_classification import classification
import pandas as pd
import numpy as np
import re
from grabber import *  # This imports your common setup code from grabber.py


import json
from config.base_config import BaseConfig
from utils.util import *
from utils.util_colors_drawing import *
from utils.util_fig_display import *
# from utils.const import *
from preprocessing import filter
from preprocessing import slide
from preprocessing import tiles
from stat_analysis import deep_cell_map
from segmentation_classification import segmentation
from segmentation_classification.classification import ModelClassification, segment_classify_cells_wsi
# from region_of_interest import Roi
from config.dataset_immucan_if1 import ImmuCanIF1Config

from stat_analysis.colocalisation_analysis import ColocAnalysis
from stat_analysis.dbscan_analysis import DbscanAnalysis
from stat_analysis.neighbours_analysis import NeighborsAnalysis
from config.datase_immucan_1140_config import ImmuCan1140Config
import os
import gzip
import pandas as pd
import os, gzip
import pandas as pd

#from config.dataset_management import take_config_from_dataset

# %matplotlib inline 


"""
# here we try to enrich the original tsv of the qptiff with more information
# final csv inclue thses columns:cell.ID: 29
# nucleus.x_from_df1: 7384.2
# nucleus.y_from_df1: 35592.2
# CD15.score: 0.539248200245906
# CK.score: 0.552554851372081
# CD3.score: 4.64166207403361
# CD11c.score: 0.219289123282272
# CD20.score: 0.145625005650925
# CD163.score: 1.32032298268653
# CD15.score.normalized: 0.1257
# CK.score.normalized: 0.2705
# CD3.score.normalized: 0.731
# CD11c.score.normalized: 0.0877
# CD20.score.normalized: 0.0725
# CD163.score.normalized: 0.278
# tissue.type: stroma
# phenotype: CD15-CK-CD3-CD11c-CD20-CD163-
# in.ROI.next_to_tumor_tissue: False
# in.ROI.tumor_tissue: True
# phenotype_norm: CD15--CK--CD3--CD11C--CD20--CD163-
# cell_type: 9
# cell_type_name: other
# B: 0
# BnT: 0
# DC: 0
# MacCD163: 0
# Neutrophil: 0
# T: 0
# Tumor: 0
# Tumor_CD15: 0
# other: 1
# nucleus.x_from_df2: 7384.2
# nucleus.y_from_df2: 35592.2
# nucleus.x.px: 11023
# nucleus.y.px: 11827
# cell.area.px: 725.0
# id_cell: 29
# x_img: 11023
# y_img: 11827
# tile_col: 10
# tile_row: 11
# tile_cat: row_11_col_10
# x_tile: 783
# y_tile: 563
# x_tile_border: 11024
# y_tile_border: 11828
# length_max: 725.0
# size: 725.0
# channel_number: 9
# """



def trace(func):
    def wrapper(*args, **kwargs):
        print(f"▶️  Entering: {func.__qualname__}")
        result = func(*args, **kwargs)
        print(f"✅ Exiting: {func.__qualname__}")
        return result
    return wrapper

class qptiff_codes:
    MARKER_ORDER = ["CD15", "CK", "CD3", "CD11C", "CD20", "CD163"]
    def __init__(self,dataset_config):
        
        self.dataset_config = dataset_config

        self.qptiff_path = None
        self.tsv_path_old = None
        self.tsv_path_nucleus = None

        self.bc_code = None
        self.if_code = None
        self.slide_id = None
        self.image_shape_H = None
        self.image_shape_W = None
        
        self.Full_CSV= None # the holder of our compelete tsv file after adding extra columns

        #self.dataset_config.qptiff_dir = dataset_config.qptiff_dir
        #self.dataset_config.tsv_dir = dataset_config.tsv_dir
        

        



    



    @trace
    def scan_folder_for_images(self):
        """
        Scan one cohort folder that contains:
            cohort_dir/
                qptiff/
                tsv/

        Returns:
            A list of entries, each describing one image with:
                - qptiff
                - phenotype TSV
                - nucleus-px TSV
        """
        cohort_dir = self.dataset_config.cohort_dir
        print(f"\n Scanning folder: {cohort_dir}")

        qptiff_dir = os.path.join(cohort_dir, "qptiff")
        tsv_dir = os.path.join(cohort_dir, "tsv")

        if not os.path.isdir(qptiff_dir):
            raise FileNotFoundError(f"ERROR: Missing folder: {qptiff_dir}")

        if not os.path.isdir(tsv_dir):
            raise FileNotFoundError(f"ERROR: Missing folder: {tsv_dir}")

        # List all qptiff files
        qptiff_files = [f for f in os.listdir(qptiff_dir) if f.endswith(".qptiff")]
        print(f"Found {len(qptiff_files)} qptiff images")

        results = []

        for qf in qptiff_files:
            # Extract image ID before "_#_"
            # Example:
            # IMMU-BC1-0191-FIXT-03-IF1-02_#_<hash>.qptiff
            # → image_id = IMMU-BC1-0191-FIXT-03-IF1-02
            image_id = qf.split("_#_")[0]
            qptiff_path = os.path.join(qptiff_dir, qf)

            # Find TSV files belonging to this image
            related_tsvs = [t for t in os.listdir(tsv_dir) if image_id in t]

            tsv_old = None
            tsv_nucleus = None

            # Inspect TSVs to detect type
            for tsv in related_tsvs:
                full_path = os.path.join(tsv_dir, tsv)

                # Open .tsv or .tsv.gz
                try:
                    if full_path.endswith(".gz"):
                        df = pd.read_csv(gzip.open(full_path, "rt"), sep="\t", nrows=5)
                    else:
                        df = pd.read_csv(full_path, sep="\t", nrows=5)
                except Exception as e:
                    print(" Failed to read TSV:", full_path, e)
                    continue

                cols = df.columns

                # TSV type: phenotype TSV
                if "phenotype" in cols:
                    tsv_old = full_path

                # TSV type: nucleus coordinate TSV
                if "nucleus.x.px" in cols:
                    tsv_nucleus = full_path

            # Save only if both TSVs exist
            if tsv_old and tsv_nucleus:
                results.append({
                    "image_id": image_id,
                    "qptiff": qptiff_path,
                    "tsv_old": tsv_old,
                    "tsv_nucleus": tsv_nucleus
                })
                print(f"Added {image_id}")
            else:
                print(f" Missing TSVs for {image_id}")

        print(f"\n Completed. Valid entries: {len(results)}")
        return results
    
    
    
    @trace
    def load_image(self, index=None):
        """
        Load image information in two modes:

        mode = "single"
            - single_image_name must be provided
            - returns {image_id, qptiff, tsv_old, tsv_nucleus}

        mode = "all"
            - dataset_list must be a list generated by scan_folder_for_images()
            - index is which image to load (0,1,2,...)
            - returns the dataset_list[index]
            - returns None when index >= len(dataset_list)
        """
        mode = self.dataset_config.processing_mode
        cohort_dir = self.dataset_config.cohort_dir
        dataset_list = self.scan_folder_for_images()
        single_image_name = self.dataset_config.single_image_name
        # -----------------------------
        # MODE 1: single image
        # -----------------------------
        if mode == "single":
            if not single_image_name:
                raise ValueError("single_image_name must be provided for single mode")

            # Build full path
            qptiff_path = os.path.join(cohort_dir, "qptiff", single_image_name)

            if not os.path.isfile(qptiff_path):
                raise FileNotFoundError(f"QPTIFF not found: {qptiff_path}")

            # Extract image ID
            image_id = single_image_name.split("_#_")[0]

            # Find corresponding TSV files
            tsv_dir = os.path.join(cohort_dir, "tsv")
            tsv_files = [t for t in os.listdir(tsv_dir) if image_id in t]

            tsv_old = None
            tsv_nucleus = None

            for t in tsv_files:
                full_path = os.path.join(tsv_dir, t)

                try:
                    df = pd.read_csv(
                        gzip.open(full_path, "rt") if full_path.endswith(".gz") else full_path,
                        sep="\t",
                        nrows=5
                    )
                except:
                    continue

                cols = df.columns

                if "phenotype" in cols:
                    tsv_old = full_path
                if "nucleus.x.px" in cols:
                    tsv_nucleus = full_path

            if not tsv_old or not tsv_nucleus:
                raise RuntimeError(f"TSV missing for {image_id}")

            self.image_id = image_id
            self.qptiff_path = qptiff_path
            self.tsv_old = tsv_old
            self.tsv_nucleus = tsv_nucleus
            
            return {
                "image_id": image_id,
                "qptiff": qptiff_path,
                "tsv_old": tsv_old,
                "tsv_nucleus": tsv_nucleus
            }

        # -----------------------------
        # MODE 2: all images in folder
        # -----------------------------
        elif mode == "all":
            if dataset_list is None:
                raise ValueError("dataset_list must be provided for all mode")

            if index >= len(dataset_list):
                return None   # finished

                # Get the dictionary for this image
                info = dataset_list[index]

                # Set class attributes instead of returning it
                self.image_id    = info["image_id"]
                self.qptiff_path = info["qptiff_path"]
                self.tsv_old     = info["tsv_old"]
                self.tsv_nucleus = info["tsv_nucleus"]

        else:
            raise ValueError("Mode must be 'single' or 'all'")

    
    
    @trace
    #Displaying the downsampled version of the image
    def display_qptiff_image(self):
        """ Display the downsampled version of the qptiff image to get an idea about the image"""
        qptiff_path3 = self.qptiff_path
        with tifffile.TiffFile(qptiff_path3) as tif:
            print(len(tif.pages))
            image = tif.pages[0].asarray(out='memmap')[::1,::1]
            rgb = tif.pages[8].asarray(out='memmap')[::1,::1]
            #plt.imshow(image,cmap='gray')
            #plt.axis('off')
            #plt.show()
            print("image shape:",image.shape)
            self.image_shape_H,self.image_shape_W = image.shape
            print("image shape:",rgb.shape)


    # not used anymore
    @trace
    def get_qptiff_files(self):
        """Here we write a code that get the link of the folders in immucan datasets 
        that contains qptiff images and tsv , the link coming from config file,self.dataset_config.qptiff_dir = dataset_config.qptiff_dir
        #self.dataset_config.tsv_dir = dataset_config.tsv_dir, the link should be point at 
        one chort at each time, like all the images in .../BC1/IF1/qptiff and BC1/IF1/tsv
        then the code run on the first image in the folder and save important information such as           
            self.bc_code =   # BC1
            self.slide_id =  # 0213
            self.if_code =  # IF1
            that is the unique info of each image, then it saves the link of that image in self.qptiff_path 
            then with this info it search for the corresponding tsv of this image , since we have 
            multiple tsv for each image , it check if the tsc contain columns name "phenotype_norm"
            if it has it saved the link of that tsv in self.tsv_path_old and if the tsv contain column with name
            "nucleus.x.px" n it save the link of that tsv in self.tsv_path_nucleus, also if there is 2 similar image exisit it ignores the second one
            
        """

        # -----------------------------------------------
        # 1. FIND THE QPTIFF IMAGE AND EXTRACT IDENTIFIERS
        # -----------------------------------------------
        qptiff_files = [
            f for f in os.listdir(self.dataset_config.qptiff_dir)
            if f.lower().endswith(".qptiff")
        ]
        if not qptiff_files:
            raise ValueError("No .qptiff files found")

        # parse naming pattern
        info_list = []
        for f in qptiff_files:
            parts = f.split("-")
            # IMMU | BC1 | 0213 | FIXT | 01 | IF1 | 01_#_xxxx.qptiff
            bc_code = parts[1]        # BC1
            slide_id = parts[2]       # 0213
            if_code = parts[5]        # IF1
            info_list.append((slide_id, bc_code, if_code, f))

        # sort → keep the FIRST file, ignore second duplicates
        info_list_sorted = sorted(info_list, key=lambda x: x[-1])
        slide_id, bc_code, if_code, filename = info_list_sorted[0]

        self.slide_id = slide_id
        self.bc_code = bc_code
        self.if_code = if_code
        self.qptiff_path = os.path.join(self.dataset_config.qptiff_dir, filename)

        print(f" Selected QPTIFF for slide {slide_id}:")
        print("   ", self.qptiff_path)

        # -----------------------------------------------
        # 2. SEARCH IN TSV DIRECTORY FOR RELATED FILES
        # -----------------------------------------------
        tsv_files = [
            f for f in os.listdir(self.dataset_config.tsv_dir)
            if (f.endswith(".tsv") or f.endswith(".tsv.gz"))
            and (bc_code in f)
            and (if_code in f)
            and (slide_id in f)
        ]

        if not tsv_files:
            raise ValueError(f"No TSV/TSV.GZ found for {bc_code}-{slide_id}-{if_code}")

        # -----------------------------------------------
        # 3. CHECK COLUMNS AND ASSIGN TO RIGHT VARIABLES
        # -----------------------------------------------
        for f in tsv_files:
            full_path = os.path.join(self.dataset_config.tsv_dir, f)

            # Read first 5000 lines max, to avoid large load
            if f.endswith(".gz"):
                reader = gzip.open(full_path, "rt")
            else:
                reader = open(full_path, "r")

            try:
                df_head = pd.read_csv(reader, sep="\t", nrows=20)
            finally:
                reader.close()

            cols = df_head.columns

            if "phenotype_norm" in cols and self.tsv_path_old is None:
                self.tsv_path_old = full_path
                print(f" phenotype_norm TSV found: {full_path}")

            if "nucleus.x.px" in cols and self.tsv_path_nucleus is None:
                self.tsv_path_nucleus = full_path
                print(f" nucleus.x.px TSV found: {full_path}")

            if self.tsv_path_old and self.tsv_path_nucleus:
                break

        # -----------------------------------------------
        # 4. ENSURE BOTH FILES FOUND OR RAISE ERROR
        # -----------------------------------------------
        if self.tsv_path_old is None:
            raise ValueError(f"No TSV with phenotype_norm for slide {slide_id}")

        if self.tsv_path_nucleus is None:
            raise ValueError(f"No TSV with nucleus.x.px for slide {slide_id}")

        print(" All required TSV files found successfully.")


    @trace
    def get_qptiff_files(self):
        """
        ONE method to scan all folders and set all 4 file paths:
            - QPTIFF (.qptiff)
            - TSV.GZ (.tsv.gz)
            - TSV (.tsv)
            - CSV (.csv)
        """

        # Define folder → extension → variable mapping
        targets = [
            (self.dataset_config.qptiff_dir, {".qptiff"}, "QPTIFF_PATH"),
            (self.dataset_config.tsv_gz_dir, {".gz"},      "TSV_GZ_PATH"),
            (self.dataset_config.tsv_dir,    {".tsv"},     "TSV_PATH"),
            (self.dataset_config.csv_dir,    {".csv"},     "CSV_PATH"),
        ]

        # Loop through all targets
        for folder, extensions, var_name in targets:

            if not os.path.isdir(folder):
                print(f"Folder not found: {folder}")
                setattr(self, var_name, None)
                continue

            files = os.listdir(folder)

            # Filter by extension
            matched = [
                f for f in files
                if os.path.splitext(f)[1].lower() in extensions
            ]

            if not matched:
                print(f" No file with ext {extensions} in {folder}")
                setattr(self, var_name, None)
                continue

            # Choose first file (sorted alphabetically)
            first_path = os.path.join(folder, sorted(matched)[0])

            # Store inside the class
            setattr(self, var_name, first_path)

            print(f" {var_name} = {first_path}")

        print("\n All file paths set.\n")
        
        
        
    def display_metadata(self):
        pass
    
    def display_image(self):
        pass 
        
    def adding_extra_columns(self):
        pass
    
    
    
    
    
    
    
    
    
    @trace
    def concatenating_csvs(self):
        """ Immucan original tsv did not contain the pixel level coordination , after they sent another tsv with pixel level coordination so
        we had to concatenate them , we did the concatenation based on cell.ID which is unique for each cell.
        Also DeepCellMp works with different cell type names and size and length and we have cell_type which contain intergers of the cell types
        and cell_type_names which contain the cell type names so we had to map them too, 
        finally we save the final csv
        """
        
        # location of the tsv with cell types
        import pandas as pd

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

        # Load TSV files            
        path_cell_type1140 = self.tsv_path_old
        df = pd.read_csv(path_cell_type1140, sep="\t" , compression="gzip")
        df2 = pd.read_csv(self.tsv_path_nucleus, sep="\t")
        #df3 = pd.read_csv("/root/cloud-data/eu-ngs/U1086086/Ali_A/CSV_concate/IMMU-BC1-1140-FIXT-01-IF1-01_#_cells_properties_#_5668e16b0454f7c408b2b42f0fb8274a.tsv.gz", sep="\t" , compression="gzip")

        # print ("df2 columns:",df2.columns)
        # print("df columns:",df.columns)

        print (df.head())
        print (df2.head())

        # Copy to a new column with a different name
        df['cell_type_name'] = df['cell_type']




        # build a two-way dict
        both = {**label_names, **{v:k for k, v in label_names.items()}}

        # # map a column that may contain keys *or* values
        df['cell_type'] = df['cell_type_name'].replace(both)   
        df["size"] = df["cell.area.px"]
        df["length"] = df["cell.area.px"]
        cats = list(label_names.values())

        # Ensure the column has names (map IDs → names; keep names as-is)
        df['cell_type_name'] = df['cell_type_name'].map(label_names).fillna(df['cell_type_name'])

        # One-hot columns named exactly as in label_names.values()
        onehot = pd.get_dummies(df['cell_type_name']).reindex(columns=cats, fill_value=0).astype('int8')
        df = df.join(onehot)


        # Concatenate
        df4 = pd.merge(
            df,
            df2,
            on="cell.ID",
            how="outer",                # keep all cells from both
            suffixes=("_from_df1", "_from_df2")  # differentiate overlapping columns
        )

        print(df4.columns)
        print(df4.head())
        print(df4[['nucleus.x.px', 'nucleus.y.px','cell.area.px']].agg(['min', 'max']))

        df4.to_csv(self.Full_CSV, sep=";", index=False)



        
    @trace      
    def calculating_celltypes_from_phenotypes(self):
        
        """ Based on the phenotype column in the original tsv we map the phenotypes to cell types based on
        a mapping csv file that we received from immcan and you an find the file in the supplemntary 2 files, here is the code 
        for BC1_IF1 cohort but we can use the same code for other cohorts by changing the input and output folder 
        and the mapping csv file path"""
        


        # ----------------------------
        # 1) CONFIG
        # ----------------------------
        mapping_csv_path = "/root/cloud-data/eu-ngs/U1086086/Ali_A/DeepCellMap/Extension/tsv_files/tsv_files_with_names/Supplementary_Table_2.csv"  # This file has been provided by Immucan Concertiom
        input_folder = self.cohort_dir
        output_folder = "/root/cloud-data/eu-ngs/U1086086/Ali_A/DeepCellMap/Extension/tsv_files/tsv_files_with_names/BC1_IF1_celltype_updated"

        # The marker order expected by the CSV mapping (based on your file)
        # MARKER_ORDER = ["CD15", "CK", "CD3", "CD11C", "CD20", "CD163"]

        # ----------------------------
        # 2) HELPERS
        # ----------------------------
        _token_re = re.compile(r"([A-Za-z0-9]+)([+\-])")


            # ----------------------------
        # 3) LOAD MAPPING (CSV)
        # ----------------------------
        map_df = pd.read_csv(mapping_csv_path)  # columns: phenotype, celltype
        # Normalize column names and values
        map_df.columns = map_df.columns.str.strip().str.lower()
        if not {"phenotype", "celltype"}.issubset(set(map_df.columns)):
            raise ValueError("Mapping CSV must have 'phenotype' and 'celltype' columns.")

        map_df["canonical"] = map_df["phenotype"].apply(canonicalize)
        # Drop any rows that couldn't be canonicalized
        map_df = map_df.dropna(subset=["canonical", "celltype"]).copy()

        # Build dictionary: canonical phenotype -> celltype
        phenotype_to_celltype = dict(zip(map_df["canonical"], map_df["celltype"]))

        print(f" Loaded {len(phenotype_to_celltype)} phenotype→celltype rules from CSV.")

        # ----------------------------
        # 4) PROCESS TSV.GZ FILES
        # ----------------------------
        all_files = [f for f in os.listdir(input_folder) if f.endswith(".tsv.gz")]
        total_files = len(all_files)
        processed_files = 0
        print(f" Total files found: {total_files}")

        os.makedirs(output_folder, exist_ok=True)

        for filename in all_files:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read TSV from gzip
            with gzip.open(input_path, "rt") as f:
                df = pd.read_csv(f, sep="\t")
            processed_files += 1
            print(f"🔄 [{processed_files}/{total_files}] {filename}")

            if "phenotype" not in df.columns:
                print(f"   Skipped (no 'phenotype' column): {filename}")
                continue

            # Canonicalize TSV phenotypes and map to cell types
            df["phenotype_norm"] = df["phenotype"].apply(canonicalize)
            df["cell_type"] = df["phenotype_norm"].map(phenotype_to_celltype)

            # Fallback for unmatched phenotypes
            df["cell_type"] = df["cell_type"].fillna("Unknown")

            # If you want to fallback to your old rule-based function for unknowns, uncomment:
            # unknown_mask = df["cell_type"].eq("Unknown") & df["phenotype"].notna()
            # df.loc[unknown_mask, "cell_type"] = df.loc[unknown_mask, "phenotype"].apply(classify_cell)

            # Write back to gzip TSV
            with gzip.open(output_path, "wt") as f_out:
                df.to_csv(f_out, sep="\t", index=False)

            print(f"  ✅ Saved with cell_type → {output_path}")

        print(f"\n Summary: {processed_files}/{total_files} files processed.")

            
        
        
        
    @trace
    def parse_phenotype(self,raw: str):
        """
        Helper for the function calculating_celltypes_from_phenotypes
        Parse strings like 'CD15-CK-CD3-CD11c-CD20-CD163+' into a dict:
        {'CD15':'-', 'CK':'-', 'CD3':'-', 'CD11C':'-', 'CD20':'-', 'CD163':'+'}
        Works even if there are spaces/underscores or inconsistent cases.
        """
        if pd.isna(raw):
            return {}
        s = str(raw).strip()
        # Tolerate spaces and underscores
        s = s.replace(" ", "").replace("_", "")
        # Make case-insensitive and unify 'CD11c' -> 'CD11C'
        pairs = _token_re.findall(s.upper())
        # pairs is list of (MARKER, SIGN)
        d = {}
        for m, sign in pairs:
            # normalize known aliases if needed (e.g., CD11c vs CD11C already uppercased)
            d[m] = sign
        return d
    @trace
    def canonicalize(self,raw: str, marker_order=MARKER_ORDER):
        """
        Helper for the function calculating_celltypes_from_phenotypes
        Convert any phenotype string to a canonical form with a fixed marker order:
            'CD15-CK-CD3-CD11C-CD20-CD163+' (no spaces, all upper-case).
        Missing markers are omitted (rare). For matching, we require all markers that
        the mapping expects; if any are missing, we still build from what's present.
        """
        d = parse_phenotype(raw)
        if not d:
            return None
        parts = []
        for m in marker_order:
            if m in d:
                parts.append(f"{m}{d[m]}")
            else:
                # If you *require* all markers, you could return None here.
                # return None
                # Otherwise, just skip missing markers.
                pass
        return "-".join(parts) if parts else None

   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    @trace 
    def calculatng_tiles(self):
        
        """ Based on the pixel level coordination we calculate the tile row and column and the local coordination inside each tile Since
        DeepCellMap works on tiles of 1024x1024 size and then we calculate the location of each cell inside each tiles since 
        DeepCellMap needs the location of the cells inside each tiles, also the tiles in the DeepCellMap are 1-indexed 
        not 0-indexed so we add +1 to the local coordination inside each tile ,also we calculate the border coordination for each cell which is global +1 px (clipped to image bounds).
        and since DeepCellMap recognize x_img and y_img as the global coordination we map the nucleus.y.px and nucleus.x.px to y_img and x_img
        finally we save the final csv with all the extra columns"""
        
        


        # ---- CONFIG ----
        file_in   = self.Full_CSV
        file_out  = self.Full_CSV


        # Image & tiles
        IMG_W, IMG_H = 51840, 46080
        TILE_W, TILE_H = 1024, 1024

        # Coordinate scale (set True only if nucleus.*.px are stored at 1/8 resolution)
        COORDS_ARE_DOWNSAMPLED = False
        SCALE_FACTOR = 8

        # ---------- READ (keep everything as-is) ----------
        df = pd.read_csv(file_in, sep=";")

        # ---------- NEW: id_cell (keep original cell.ID too) ----------
        # Use nullable Int64 to keep NA where parsing fails
        if "cell.ID" in df.columns:
            df["id_cell"] = pd.to_numeric(df["cell.ID"], errors="coerce").astype("Int64")
        else:
            raise KeyError("Column 'cell.ID' not found in input")

        # ---------- NEW: x_img / y_img (from nucleus.*.px; keep originals) ----------
        for src, dst in [("nucleus.x.px", "x_img"), ("nucleus.y.px", "y_img")]:
            if src not in df.columns:
                raise KeyError(f"Column '{src}' not found in input")
            vals = pd.to_numeric(df[src], errors="coerce")
            if COORDS_ARE_DOWNSAMPLED:
                vals = (vals * SCALE_FACTOR)
            df[dst] = vals.round().clip(lower=0)  # clip lower at 0; we'll upper-clip below
        df["x_img"] = df["x_img"].clip(upper=IMG_W - 1).astype("Int64")
        df["y_img"] = df["y_img"].clip(upper=IMG_H - 1).astype("Int64")

        # ---------- NEW: tile indices & local coordinates ----------
        df["tile_col"] = (df["x_img"] // TILE_W).clip(0, (IMG_W - 1)//TILE_W).astype("Int64")
        df["tile_row"] = (df["y_img"] // TILE_H).clip(0, (IMG_H - 1)//TILE_H).astype("Int64")
        df["tile_cat"] = "row_" + df["tile_row"].astype(str) + "_col_" + df["tile_col"].astype(str)

        df["x_tile"] = (df["x_img"] % TILE_W).astype("Int64") + 1
        df["y_tile"] = (df["y_img"] % TILE_H).astype("Int64") + 1

        # Border = global +1 px (clipped to image bounds). If you prefer "inside tile", change to x_tile/y_tile + 1.
        df["x_tile_border"] = np.minimum(df["x_img"] + 1, IMG_W - 1).astype("Int64")
        df["y_tile_border"] = np.minimum(df["y_img"] + 1, IMG_H - 1).astype("Int64")

        # ---------- NEW: length_max placeholder ----------
        if "length_max" not in df.columns:
            df["length_max"] = pd.NA  # compute later from segmentation if available

        # ---------- NEW: one-hot probabilities per cell_type ----------
        # # We keep the original 'cell_type' column and create proba_* columns from it.
        # ct = df.get("cell_type", pd.Series(index=df.index, dtype="object")).astype(str).str.strip()

        # # Count and print distinct labels (helps you see what proba_* columns you'll get)
        # counts = ct.value_counts(dropna=False)
        # print("\nDistinct cell_type counts:\n", counts.to_string(), "\n")

        # Helper to sanitize class names into safe column suffixes
        def safe_name(self,s: str) -> str:
            s = s.strip().lower()
            s = s.replace("+", "pos").replace("-", "neg").replace("/", "_")
            s = re.sub(r"[^\w]+", "_", s)     # non-alnum -> _
            s = re.sub(r"_+", "_", s).strip("_")
            return s or "unknown"

        # # Build dummies then sanitize their column names
        # dummies = pd.get_dummies(ct, prefix="proba", prefix_sep="_").astype("int8")
        # dummies.columns = [
        #     "proba_" + safe_name(col.split("_", 1)[1]) if "_" in col else "proba_" + safe_name(col)
        #     for col in dummies.columns
        # ]

        # # Attach to main df (this keeps all original columns)
        # df = pd.concat([df, dummies], axis=1)

        # print("Created probability columns:", [c for c in df.columns if c.startswith("proba_")])


        # df.rename(columns={"proba_b": "B"}, inplace=True)
        # df.rename(columns={"proba_bnt": "BnT"}, inplace=True)
        # df.rename(columns={"proba_dc": "DC"}, inplace=True)
        # df.rename(columns={"proba_maccd163": "MacCD163"}, inplace=True)
        # df.rename(columns={"proba_neutrophil": "Neutrophil"}, inplace=True)
        # df.rename(columns={"proba_t": "T"}, inplace=True)
        # df.rename(columns={"proba_tumor": "Tumor"}, inplace=True)
        # df.rename(columns={"proba_tumor_cd15": "Tumor_CD15"}, inplace=True)
        # df.rename(columns={"proba_other": "other"}, inplace=True)


        # ---------- SAVE: keep ALL original columns + ALL new columns ----------
        df.to_csv(file_out, index=False)
        print(f"\nSaved: {file_out}\nRows: {len(df)}  Cols: {df.shape[1]}")
        print(df.columns.tolist())
        
        
        
        
        
        
        
        
        
        
        
        
        
    @trace
    def calculating_binary_celltypes(self):
        """ DeepCellMap contains columns with the name of the cell types which contains cell type name and in the location 
        of the cell type name we have 1 if the cell belong to that type and 0 if not
            so  we create a dataframe with a column called label_id which contain some intergers from 1 to 9 
            then
        Based on the cell_type column which contain intergers from 1 to 9 we map them to their names"""
    

        df5 = pd.DataFrame({'label_id': [1, 2, 7, 4, 9]})
        # Mapping dictionary
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

        print(df5.columns)

        # build a two-way dict
        both = {**label_names, **{v:k for k, v in label_names.items()}}

        # # map a column that may contain keys *or* values
        df5['mapped'] = df5['label_id'].replace(both)   
        #df5['new_one'] = df5['mapped'].replace(both)     



        cats = list(label_names.values())

        # Ensure the column has names (map IDs → names; keep names as-is)
        df5['mapped'] = df5['mapped'].map(label_names).fillna(df5['mapped'])

        # One-hot columns named exactly as in label_names.values()
        onehot = pd.get_dummies(df5['mapped']).reindex(columns=cats, fill_value=0).astype('int8')
        df5 = df5.join(onehot)
        print(df5.head())
        
        
        
        
        
    def get_tissue_percentage(self):
        pass


    def generate_tissue_percentage_from_mask(self,mask_path, output_csv="/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/tiles_and_tissue_percentage.csv"):
        """
        Compute tissue percentage per tile based on a binary mask image.
        1 = tissue pixel, 0 = background pixel
        by using this function you can run blew code:
        
        generate_tissue_percentage_from_mask(
            mask_path="/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/masks/IMMU-BC1-1140-FIXT-01-IF1-01_#_0296bfe21068affb025b0bea8706cc98_mask.png",
            tile_size=1024,
            #output_csv="/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/tiles_and_tissue_percentage.csv"
            output_csv= "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/classified_images/slide_001/tiles_and_tissue_percentage.csv"
        )
        """
        tile_size=224
        Image.MAX_IMAGE_PIXELS = None
        # --- 1. Load mask image ---
        mask = np.array(Image.open(mask_path))
        mask = np.where(mask > 0, 1, 0)  # convert to binary (1 tissue, 0 background)
        
        n_rows = mask.shape[0] // tile_size + 1
        n_cols = mask.shape[1] // tile_size + 1
        
        data = []
        
        # --- 2. Loop over tiles and compute tissue fraction ---
        for r in range(n_rows):
            for c in range(n_cols):
                tile = mask[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size]
                if tile.size == 0:
                    continue
                tissue_percentage = np.sum(tile) * 100.0 / tile.size
                
                # classify the tile location
                if r == 0:
                    if c == 0:
                        cat = "top-left"
                    elif c == n_cols - 1:
                        cat = "top-right"
                    else:
                        cat = "top"
                elif r == n_rows - 1:
                    if c == 0:
                        cat = "bottom-left"
                    elif c == n_cols - 1:
                        cat = "bottom-right"
                    else:
                        cat = "bottom"
                elif c == 0:
                    cat = "left"
                elif c == n_cols - 1:
                    cat = "right"
                else:
                    cat = "center"
                
                data.append([r + 1, c + 1, cat, tissue_percentage])
        
        df = pd.DataFrame(data, columns=["tile_row", "tile_col", "categorie", "tissue_percentage"])
        df.to_csv(output_csv, sep=";", index=False)
        print(" Saved:", output_csv)
        return df
    
    
    
    def Calculate_mask_cell_qptiff(self):
        """ Based on the pixel level coordination form the final csv and based on the cell type columns we create a mask for each cell type
        so each cell type will have a unique color in the mask image"""
        import pandas as pd
        import numpy as np
        import tifffile
        import math
        import pprint
        import plotly.express as px
        import plotly.graph_objects as go

        qptiff_path3 = self.qptiff_path
        mask = "/root/cloud-data/eu-ngs/U1086086/Test2/DeepCellMap/output/NewDataset3/masks/IMMU-BC1-1140-FIXT-01-IF1-01_#_0296bfe21068affb025b0bea8706cc98_mask.png"
        info_data = self.Full_CSV 
        # Map numbers to cell-type names
        LABEL_NAMES = {
            1: "B",
            2: "BnT",
            3: "DC",
            4: "MacCD163",
            5: "Neutrophil",
            6: "T",
            7: "Tumor",
            8: "Tumor_CD15",
            9: "other"
        }
        #Displaying the downsampled version of the image
        with tifffile.TiffFile(qptiff_path3) as tif:
            image = tif.pages[0].asarray(out='memmap')[::1,::1]
            a= image.shape
            print("image shape:",a)
            s = np.zeros((a[0], a[1]), dtype=np.uint16)
            df = pd.read_csv(info_data)     
            for i in range(100):
                
                x = df.iloc[i]['nucleus.x.px']
                y = df.iloc[i]['nucleus.y.px']
                area= df.iloc[i]['cell.area.px']
                r = math.sqrt(area/math.pi)	 

                if df.iloc[i]['B'] == 1:
                            for j in range(int(x-r), int(x+r)):
                                for k in range(int(y-r), int(y+r)):
                                    if math.sqrt((x-j)**2 + (y-k)**2) <= r:
                                        if (k<a[0]) and (j<a[1]) and (k>=0) and (j>=0):
                                            s[k,j]=1
                elif df.iloc[i]['BnT'] == 1:
                            for j in range(int(x-r), int(x+r)):
                                for k in range(int(y-r), int(y+r)):
                                    if math.sqrt((x-j)**2 + (y-k)**2) <= r:
                                        if (k<a[0]) and (j<a[1]) and (k>=0) and (j>=0):
                                            s[k,j]=2
                elif df.iloc[i]['DC'] == 1:
                            for j in range(int(x-r), int(x+r)):
                                for k in range(int(y-r), int(y+r)):
                                    if math.sqrt((x-j)**2 + (y-k)**2) <= r:
                                        if (k<a[0]) and (j<a[1]) and (k>=0) and (j>=0):
                                            s[k,j]=3
                elif df.iloc[i]['MacCD163'] == 1:
                            for j in range(int(x-r), int(x+r)):
                                for k in range(int(y-r), int(y+r)):
                                    if math.sqrt((x-j)**2 + (y-k)**2) <= r:
                                        if (k<a[0]) and (j<a[1]) and (k>=0) and (j>=0):
                                            s[k,j]=4 
                elif df.iloc[i]['Neutrophil'] == 1:
                            for j in range(int(x-r), int(x+r)):
                                for k in range(int(y-r), int(y+r)):
                                    if math.sqrt((x-j)**2 + (y-k)**2) <= r:
                                        if (k<a[0]) and (j<a[1]) and (k>=0) and (j>=0):
                                            s[k,j]=5
                elif df.iloc[i]['T'] == 1:
                            for j in range(int(x-r), int(x+r)):
                                for k in range(int(y-r), int(y+r)):
                                    if math.sqrt((x-j)**2 + (y-k)**2) <= r:
                                        if (k<a[0]) and (j<a[1]) and (k>=0) and (j>=0):
                                            s[k,j]=6 
                elif df.iloc[i]['Tumor'] == 1:
                            for j in range(int(x-r), int(x+r)):
                                for k in range(int(y-r), int(y+r)):
                                    if math.sqrt((x-j)**2 + (y-k)**2) <= r:
                                        if (k<a[0]) and (j<a[1]) and (k>=0) and (j>=0):
                                            s[k,j]=7 
                elif df.iloc[i]['Tumor_CD15'] == 1:
                            for j in range(int(x-r), int(x+r)):
                                for k in range(int(y-r), int(y+r)):
                                    if math.sqrt((x-j)**2 + (y-k)**2) <= r:
                                        if (k<a[0]) and (j<a[1]) and (k>=0) and (j>=0):
                                            s[k,j]=8 
                elif df.iloc[i]['other'] == 1:
                            for j in range(int(x-r), int(x+r)):
                                for k in range(int(y-r), int(y+r)):
                                    if math.sqrt((x-j)**2 + (y-k)**2) <= r:
                                        if (k<a[0]) and (j<a[1]) and (k>=0) and (j>=0):
                                            s[k,j]=9       
        print(s)
        count_non_zero = np.count_nonzero(s)
        print("Number of non-zero cells:", count_non_zero)

        fig_image = px.imshow(s, text_auto=True, title="2D NumPy Array as Image")
        fig_image.show()

        result = {}
        for num, name in LABEL_NAMES.items():
            result[name] = (s == num)
            mask2 = result[name]
        print("\nDictionary Analysis:")
        pprint(result)    
        for key, arr in result.items():
            analysis = {
                'length': mask.size,
                'non_zero_elements': np.count_nonzero(mask),
                'non_nan_elements': np.count_nonzero(~np.isnan(mask))
            }
            print(f"{key}: {analysis}")

        return result
    
    
#############################################################################################
############################################################################################
#############################################################################################
    # """code from file region of interests that have been modified or recreated, here the original functions
    # are besides the new or modified one to show the same application for easier comparision  """
            
    
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
    def get_mask_tissue(self):
        # New
        """New function"""
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    ################################################################################################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    @trace
    def get_mask_tissue_CJ(self, af_channel=7, min_obj_size=500, hole_size=500, show_plots=True):
        # New
        """
        Build aligned tissue mask directly from the QPTIFF autofluorescence channel.
        Produces a mask matching DeepCellMap ROI with borders. /
        
        / not used , to test if getting mask from qptiff channel is possible
        """

        import tifffile
        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.filters import threshold_otsu
        from skimage.morphology import remove_small_holes, remove_small_objects

        # ===============================
        # 1) Read FULL-RES autofluorescence from QPTIFF
        # ===============================
        qptiff_path = self.qptiff_path
        print(" Loading QPTIFF:", qptiff_path)

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
        print(" Applying Otsu threshold…")
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

        print(" ROI boundaries:")
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

        print(" Tissue mask stored in self.mask_tissue_w_borders")

        # Extra debugging
        print("Mask dtype:", mask_roi.dtype)
        print("Mask unique values:", np.unique(mask_roi))

        return mask_roi   
       
       
       
       
       
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
        # New
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
        qptiff_path = self.qptiff_path
        with tifffile.TiffFile(qptiff_path) as tif:
            # Load pages 0 to 8
            for page_i in range(6):
                arr = tif.pages[page_i].asarray(out='memmap')
                img_dict[page_i] = arr

                # # If page 8 is RGB → also store under "RGB"
                # if page_i == 8 and arr.ndim == 3 and arr.shape[-1] == 3:
                #     img_dict["RGB"] = arr
                    
        # ROI limits
        x_min, x_max, y_min, y_max, _, _, _, _ = self._get_roi_w_borders_lims()

        ROI_dict = {}

        for key, img in img_dict.items():

            # Grayscale pages → shape (H, W)
            if img.ndim == 2:
                ROI_dict[key] = img[y_min:y_max, x_min:x_max]

            # # RGB page → shape (H, W, 3)
            # elif img.ndim == 3:
            #     cut_dict[key] = img[y_min:y_max, x_min:x_max, :] 
        self.image_w_borders = ROI_dict              

        return ROI_dict




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
        H = self.image_shape_H
        W = self.image_shape_W
        x_min = max(0, min(x_min, H))
        x_max = max(0, min(x_max, H))
        y_min = max(0, min(y_min, W))
        y_max = max(0, min(y_max, W))

        x_min_borders = max(0, min(x_min_borders, H))
        x_max_borders = max(0, min(x_max_borders, H))
        y_min_borders = max(0, min(y_min_borders, W))
        y_max_borders = max(0, min(y_max_borders, W))
            
        
        return x_min, x_max, y_min, y_max, x_min_borders, x_max_borders, y_min_borders, y_max_borders
















        ###################################################################
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
    @trace
    def get_table_cells(self, channels_of_interest = None, save = False,filter_cells_from_several_channels=True):
        # Modified , *lines commented and 2 lines added to make this function read from the TSV
        
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

    def _get_table_cells_fluorescence(self, channels_of_interest="all", filter_cells_from_several_channels=True):
        
        
        
        """
        Get the table of the cells for fluorescence images

        Test if done on the entire image and do it on this particular ROI if not
        """
        # #Test if done for entire slide 
        # path_classified_slide = os.path.join(self.dataset_config.dir_classified_img, "slide_{}_cells.csv".format(self.slide_num))
        particular_roi = (self.origin_row, self.origin_col, self.end_row, self.end_col)
        path_classified_slide = classification.get_path_table_cells(self.slide_num,self.dataset_config,particular_roi = None)

        path_classified_roi = classification.get_path_table_cells(self.slide_num,self.dataset_config,particular_roi = particular_roi)
    
    # modification to read from qptiff tsv
    
        # if os.path.exists(path_classified_slide):
        #     table_cells = pd.read_csv(path_classified_slide,sep = ";")

        # else : #If not, compute it on ROI only 
        #     particular_roi = (self.origin_row, self.origin_col, self.end_row, self.end_col)
        #     print("particular_roi",particular_roi)
        #     table_cells = classification.segment_classify_cells(self.slide_num, self.dataset_config,channels_cells_to_segment=channels_of_interest, particular_roi = particular_roi)
        #     table_cells.to_csv(path_classified_roi, sep = ";", index = False)


        custom_csv = self.Full_CSV
        table_cells = pd.read_csv(custom_csv, sep=";")

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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    #################################################################################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
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
    def _get_cell_masks_fluorescence(self, channels_of_interest=None):
        qptiff_path = self.qptiff_path
        info_csv = self.Full_CSV
        
        # Must return the dict keyed by class names ("B","T",...)


        return self.Calculate_mask_cell_qptiff(qptiff_path, info_csv)
    
    
    
    
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











######################################################################################################















    # in the slide file changed the path to read from the tsv in order 
    @trace
    def _get_cells_statistics(dict_statistics_images, dataset_config, image_number):
        # modified
        """
        Create the dictionnary with all the stats from table_cells 
        """
        #path_table_cells = util.get_path_table_cells(image_number,dataset_config)
        path_table_cells = self.Full_CSV
        """
        if os.path.exists(path_table_cells):
            table_cells_slide = pd.read_csv(path_table_cells, sep = ";")
        else :
            raise Exception("Table cells not found -> all tiles hasn't been segmented and classified")
        """
        table_cells_slide = pd.read_csv(path_table_cells, sep = ";")
        print("Columns in CSV:", table_cells_slide.columns.tolist())
        dict_statistics_images["n_cells_slide"] = len(table_cells_slide)
        dict_statistics_images["mean_n_cells_per_tile_slide"] = table_cells_slide.groupby(['tile_row', 'tile_col'])['id_cell'].count().mean()
        dict_statistics_images["std_n_cells_per_tile_slide"] = table_cells_slide.groupby(['tile_row', 'tile_col'])['id_cell'].count().std()
        dict_statistics_images["mean_cell_size_slide"] = table_cells_slide['size'].mean()
        dict_statistics_images["std_cell_size_slide"] = table_cells_slide['size'].std()

        for idx_decision, cell_type_name in enumerate(dataset_config.cell_class_names,1):
            print("idx_decision", idx_decision)
            sub_table_cells_slide = table_cells_slide[table_cells_slide["cell_type"] == idx_decision]
            print("sub_table_cells_slide", sub_table_cells_slide.shape)
            dict_statistics_images["n_cells_{}_slide".format(cell_type_name)] = table_cells_slide[table_cells_slide["cell_type"] == idx_decision].shape[0]
            dict_statistics_images["fraction_{}_slide".format(cell_type_name)] = table_cells_slide[table_cells_slide["cell_type"] == idx_decision].shape[0]/dict_statistics_images["n_cells_slide"]
            dict_statistics_images["mean_size_{}_slide".format(cell_type_name)] = table_cells_slide[table_cells_slide['cell_type'] == idx_decision]['size'].mean()
            dict_statistics_images["std_size_{}_slide".format(cell_type_name)] = table_cells_slide[table_cells_slide['cell_type'] == idx_decision]['size'].std()
            if dataset_config.statistics_with_proba:
                dict_statistics_images["n_cells_{}_proba_slide".format(cell_type_name)] = table_cells_slide["proba_{}".format(cell_type_name)].sum()
                dict_statistics_images["fraction_{}_proba_slide".format(cell_type_name)] = table_cells_slide["proba_{}".format(cell_type_name)].sum()/dict_statistics_images["n_cells_slide"]
        # if dataset_config.data_type == "fluorescence":
        #     pass
        # elif dataset_config.data_type == "wsi": 
        #     pass
        # else : 
        #     raise Exception("Data type not found")
        return dict_statistics_images