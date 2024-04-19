from typing import List, Callable
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from PIL import Image
from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException
from pathlib import Path, PurePath


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

    def __init__(self, hostname: str = 'ml.facebase.org', catalog_id: str = 'isa', data_dir: str = './'):
        """
        Initializes the FacebaseML object.

        Args:
        - hostname (str): The hostname of the server where the catalog is located.
        - catalog_number (str): The catalog number or name.
        """

        super().__init__(hostname, catalog_id, 'isa', data_dir)
