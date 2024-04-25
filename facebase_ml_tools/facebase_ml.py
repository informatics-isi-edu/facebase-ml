from typing import List, Callable
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from PIL import Image
from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException
from pathlib import Path, PurePath
import os

# import re


class FaceBaseMLException(DerivaMLException):
    def __init__(self, msg=""):
        super().__init__(msg=msg)


class FaceBaseML(DerivaML):
    """
    FaceBaseML is a class that extends DerivaML and provides additional routines for working with Facebase
    catalogs using deriva-py.

    Attributes:
    - protocol (str): The protocol used to connect to the catalog (e.g., "https").
    - hostname (str): The hostname of the server where the catalog is located.
    - catalog_number (str): The catalog number or name.
    - credential (object): The credential object used for authentication.
    - catalog (ErmrestCatalog): The ErmrestCatalog object representing the catalog.
    - pb (PathBuilder): The PathBuilder object for constructing URL paths.

    Methods:
    - __init__(self, hostname: str = 'ml.facebase.org', catalog_number: str = 'eye-ai'): Initializes the EyeAI object.
    """

    def __init__(self, hostname: str = 'ml.facebase.org', catalog_id: str = 'fb-ml',
                 cache_dir: str='/data', working_dir: str='./'):
        """
        Initializes the FacebaseML object.

        Args:
        - hostname (str): The hostname of the server where the catalog is located.
        - catalog_number (str): The catalog number or name.
        """

        super().__init__(hostname, catalog_id, 'ml', cache_dir, working_dir)
        
    def join_and_save_csv(self, base_dir, biosample_filename, genotype_filename, output_filename):
        """
        Joins two CSV files based on specific columns and saves the result to a new file.
    
        Parameters:
        base_dir (str): The base directory path.
        biosample_filename (str): The filename for the biosample CSV.
        genotype_filename (str): The filename for the genotype CSV.
        output_filename (str): The filename to save the joined table.
        """
        # Construct full paths for the files
        biosample_path = os.path.join(base_dir, biosample_filename)
        genotype_path = os.path.join(base_dir, genotype_filename)
        output_path = os.path.join(base_dir, output_filename)
    
        # Load the CSV files
        biosample_df = pd.read_csv(biosample_path)
        genotype_df = pd.read_csv(genotype_path)
    
        # Join the tables based on the 'genotype' column of biosample and 'id' column of genotype
        merged_df = pd.merge(biosample_df, genotype_df, left_on='genotype', right_on='id')
    
        # Select and rename the required columns
        final_df = merged_df[['local_identifier', 'name']]
        final_df = final_df.rename(columns={'local_identifier':'Biosample','name': 'genotype'})

        final_df['Experimental_Group'] = final_df['genotype'].apply(lambda x: 'Control' if x.endswith('+/+') else 'Experiment')

        # Save the final dataframe to a new CSV file
        final_df.to_csv(output_path, index=False)
        return final_df, output_path
