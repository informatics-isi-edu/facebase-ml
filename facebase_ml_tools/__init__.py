from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("facebase")
except PackageNotFoundError:
    # package is not installed
    pass
