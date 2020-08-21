from imlib.general.system import ensure_directory_exists


class Paths:
    """
    A single class to hold all file paths that may be used. Any paths
    prefixed with "tmp__" refer to internal intermediate steps, and will be
    deleted if "--debug" is not used.
    """

    def __init__(self, brainreg_directory, standard_space=True):
        self.brainreg_directory = brainreg_directory

        self.main_directory = self.brainreg_directory / "manual_segmentation"
        ensure_directory_exists(self.main_directory)

        if standard_space:
            self.segmentation_directory = (
                self.main_directory / "standard_space"
            )
        else:
            self.segmentation_directory = self.main_directory / "sample_space"

        ensure_directory_exists(self.segmentation_directory)

        self.regions_directory = self.join_seg_files("regions")
        self.region_summary_csv = self.regions_directory / "summary.csv"

        self.tracks_directory = self.join_seg_files("tracks")

    def join_seg_files(self, filename):
        return self.segmentation_directory / filename
