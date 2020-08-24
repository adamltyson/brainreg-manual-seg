import numpy as np

from qtpy import QtCore
from pathlib import Path
from glob import glob

from brainrender.scene import Scene

from brainreg_manual_seg.paths import Paths

from brainreg_manual_seg.callbacks import (
    region_analysis,
    track_analysis,
    save_all,
)
from brainreg_manual_seg.man_seg_tools import (
    add_existing_region_segmentation,
    add_existing_track_layers,
    add_new_track_layer,
    add_new_region_layer,
)
from brainreg_manual_seg.gui.elements import (
    add_button,
    add_checkbox,
    add_float_box,
    add_int_box,
)

from qtpy.QtWidgets import (
    QLabel,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QWidget,
)
from bg_atlasapi.bg_atlas import BrainGlobeAtlas


def display_brain_region_name(layer, structures):
    val = layer.get_value()
    if val != 0 and val is not None:
        try:
            msg = structures[val]["name"]
        except KeyError:
            msg = "Unknown region"
    else:
        msg = "No label here!"
    layer.help = msg


class General(QWidget):
    def __init__(
        self,
        viewer,
        point_size=100,
        spline_size=50,
        track_file_extension=".h5",
        image_file_extension=".tiff",
        num_colors=10,
        brush_size=250,
        spline_points_default=1000,
        spline_smoothing_default=0.1,
        fit_degree_default=3,
        summarise_track_default=True,
        add_surface_point_default=False,
        calculate_volumes_default=True,
        summarise_volumes_default=True,
        boundaries_string="Boundaries",
    ):
        super(General, self).__init__()
        self.point_size = point_size
        self.spline_size = spline_size
        self.brush_size = brush_size

        # general variables
        self.viewer = viewer

        # track variables
        self.track_layers = []
        self.track_file_extension = track_file_extension
        self.spline_points_default = spline_points_default
        self.spline_smoothing_default = spline_smoothing_default
        self.summarise_track_default = summarise_track_default
        self.add_surface_point_default = add_surface_point_default
        self.fit_degree_default = fit_degree_default

        # region variables
        self.label_layers = []
        self.image_file_extension = image_file_extension
        self.num_colors = num_colors
        self.calculate_volumes_default = calculate_volumes_default
        self.summarise_volumes_default = summarise_volumes_default

        # atlas variables
        self.region_labels = []

        self.common_coordinate_space_default = True

        self.boundaries_string = boundaries_string
        self.setup_layout()

    def setup_layout(self):
        self.instantiated = False
        self.layout = QGridLayout()

        self.common_coordinate_space_checkbox = add_checkbox(
            self.layout,
            self.common_coordinate_space_default,
            "Analyse in common space",
            0,
        )
        self.load_button = add_button(
            "Load project",
            self.layout,
            self.load_brainreg_directory,
            1,
            0,
            minimum_width=200,
        )

        self.save_button = add_button(
            "Save", self.layout, self.save, 7, 1, visibility=False
        )

        self.status_label = QLabel()
        self.status_label.setText("Ready")

        self.layout.addWidget(self.status_label, 8, 0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(4)
        self.setLayout(self.layout)

        self.add_track_panel(self.layout)
        self.add_region_panel(self.layout)

        self.setLayout(self.layout)

    def add_track_panel(self, layout):
        self.track_panel = QGroupBox("Track tracing")
        track_layout = QGridLayout()

        add_button(
            "Add track", track_layout, self.add_track, 5, 0,
        )
        add_button(
            "Trace tracks", track_layout, self.run_track_analysis, 5, 1,
        )

        self.summarise_track_checkbox = add_checkbox(
            track_layout, self.summarise_track_default, "Summarise", 0,
        )

        self.add_surface_point_checkbox = add_checkbox(
            track_layout,
            self.add_surface_point_default,
            "Add surface point",
            1,
        )

        self.fit_degree = add_int_box(
            track_layout, self.fit_degree_default, 1, 5, "Fit degree", 2,
        )

        self.spline_smoothing = add_float_box(
            track_layout,
            self.spline_smoothing_default,
            0,
            1,
            "Spline smoothing",
            0.1,
            3,
        )

        self.spline_points = add_int_box(
            track_layout,
            self.spline_points_default,
            1,
            10000,
            "Spline points",
            4,
        )

        track_layout.setColumnMinimumWidth(1, 150)
        self.track_panel.setLayout(track_layout)
        layout.addWidget(self.track_panel, 3, 0, 1, 2)

        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        self.track_panel.setVisible(False)

    def add_region_panel(self, layout):
        self.region_panel = QGroupBox("Region analysis")
        region_layout = QGridLayout()

        add_button(
            "Add region", region_layout, self.add_new_region, 2, 0,
        )
        add_button(
            "Analyse regions", region_layout, self.run_region_analysis, 2, 1,
        )

        self.calculate_volumes_checkbox = add_checkbox(
            region_layout,
            self.calculate_volumes_default,
            "Calculate volumes",
            0,
        )

        self.summarise_volumes_checkbox = add_checkbox(
            region_layout,
            self.summarise_volumes_default,
            "Summarise volumes",
            1,
        )

        region_layout.setColumnMinimumWidth(1, 150)
        self.region_panel.setLayout(region_layout)
        layout.addWidget(self.region_panel, 5, 0, 1, 2)

        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        self.region_panel.setVisible(False)

    def initialise_image_view(self):
        self.set_z_position()

    def set_z_position(self):
        midpoint = int(round(len(self.base_layer.data) / 2))
        self.viewer.dims.set_point(0, midpoint)

    def load_brainreg_directory(self):
        self.status_label.setText("Loading...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.directory = QFileDialog.getExistingDirectory(
            self, "Select brainreg directory", options=options,
        )
        if self.directory != "":
            self.directory = Path(self.directory)
            if len(self.viewer.layers) != 0:
                # remove old layers
                for layer in list(self.viewer.layers):
                    self.viewer.layers.remove(layer)
            if self.common_coordinate_space_checkbox.isChecked():
                plugin = "brainreg_standard"
            else:
                plugin = "brainreg"
            self.viewer.open(str(self.directory), plugin=plugin)
            self.paths = Paths(
                self.directory,
                standard_space=self.common_coordinate_space_checkbox.isChecked(),
            )

            self.initialise_layers()

    def initialise_layers(self):
        # for consistency, don't load this
        try:
            self.viewer.layers.remove(self.boundaries_string)
        except KeyError:
            pass

        self.base_layer = self.viewer.layers["Registered image"]
        self.metadata = self.base_layer.metadata
        self.atlas = self.metadata["atlas_class"]
        self.atlas_layer = self.viewer.layers[self.metadata["atlas"]]

        self.reset_variables()

        self.initialise_image_view()

        @self.atlas_layer.mouse_move_callbacks.append
        def display_region_name(layer, event):
            display_brain_region_name(layer, self.atlas.structures)

        self.load_button.setMinimumWidth(0)
        self.save_button.setVisible(True)
        self.initialise_region_segmentation()
        self.initialise_track_tracing()
        self.status_label.setText("Ready")

    def reset_variables(self):
        self.mean_voxel_size = int(
            np.sum(self.atlas.resolution) / len(self.atlas.resolution)
        )
        self.point_size = self.point_size / self.mean_voxel_size
        self.spline_size = self.spline_size / self.mean_voxel_size
        self.brush_size = self.brush_size / self.mean_voxel_size

    def initialise_track_tracing(self):
        track_files = glob(
            str(self.paths.tracks_directory) + "/*" + self.track_file_extension
        )
        if self.paths.tracks_directory.exists() and track_files != []:
            for track_file in track_files:
                self.track_layers.append(
                    add_existing_track_layers(
                        self.viewer, track_file, self.point_size,
                    )
                )
        self.track_panel.setVisible(True)
        self.region_panel.setVisible(True)
        self.scene = Scene(add_root=True, atlas=self.atlas.atlas_name)
        self.splines = None

    def add_track(self):
        print("Adding a new track\n")
        add_new_track_layer(self.viewer, self.track_layers, self.point_size)

    def run_track_analysis(self):
        print("Running track analysis")
        self.scene, self.splines = track_analysis(
            self.viewer,
            self.scene,
            self.atlas,
            self.paths.tracks_directory,
            self.spline_size,
            add_surface_to_points=self.add_surface_point_checkbox.isChecked(),
            spline_points=self.spline_points.value(),
            fit_degree=self.fit_degree.value(),
            spline_smoothing=self.spline_smoothing.value(),
            point_size=self.point_size,
            spline_size=self.spline_size,
            summarise_track=self.summarise_track_checkbox.isChecked(),
            track_file_extension=self.track_file_extension,
        )
        print("Finished!\n")

    def initialise_region_segmentation(self):
        add_existing_region_segmentation(
            self.paths.regions_directory,
            self.viewer,
            self.label_layers,
            self.image_file_extension,
        )

    def add_new_region(self):
        print("Adding a new region\n")
        add_new_region_layer(
            self.viewer,
            self.label_layers,
            self.base_layer.data,
            self.brush_size,
            self.num_colors,
        )

    def run_region_analysis(self):
        print("Running region analysis")
        worker = region_analysis(
            self.label_layers,
            self.atlas_layer.data,
            self.atlas,
            self.paths.regions_directory,
            output_csv_file=self.paths.region_summary_csv,
            volumes=self.calculate_volumes_checkbox.isChecked(),
            summarise=self.summarise_volumes_checkbox.isChecked(),
        )
        worker.start()

    def save(self):
        if self.label_layers or self.track_layers:
            print("Saving")
            worker = save_all(
                self.paths.regions_directory,
                self.paths.tracks_directory,
                self.label_layers,
                self.track_layers,
                track_file_extension=self.track_file_extension,
            )
            worker.start()
