import numpy as np

from glob import glob
from pathlib import Path
from napari.qt.threading import thread_worker

from brainio.brainio import load_any
from imlib.general.system import ensure_directory_exists

from brainreg_manual_seg.man_seg_tools import (
    save_regions_to_file,
    analyse_region_brain_areas,
    summarise_brain_regions,
    analyse_track,
    analyse_track_anatomy,
    convert_and_save_points,
)


def convert_vtk_spline_to_napari_path(
    spline, x_scaling, y_scaling, z_scaling, max_z
):
    """
    Converts a vtkplotter spline object to points that can be rendered in
    napari
    :param spline: vtkplotter spline object
    :param x_scaling: scaling from image space to brainrender scene
    :param y_scaling: scaling from image space to brainrender scene
    :param z_scaling: scaling from image space to brainrender scene
    :param max_z: Maximum extent of the image in z
    :return: np.array of spline points
    """
    napari_spline = np.copy(spline.points())
    napari_spline[:, 0] = (z_scaling * max_z - napari_spline[:, 0]) / z_scaling
    napari_spline[:, 1] = napari_spline[:, 1] / y_scaling
    napari_spline[:, 2] = napari_spline[:, 2] / x_scaling
    return napari_spline.astype(np.int16)


def view_spline(
    viewer,
    image_layer,
    spline,
    x_scaling,
    y_scaling,
    z_scaling,
    spline_size,
    name="Spline fit",
):
    max_z = len(image_layer.data)
    napari_spline = convert_vtk_spline_to_napari_path(
        spline, x_scaling, y_scaling, z_scaling, max_z
    )

    viewer.add_points(
        napari_spline,
        size=spline_size,
        edge_color="cyan",
        face_color="cyan",
        blending="additive",
        opacity=0.7,
        name=name,
    )


def track_analysis(
    viewer,
    base_layer,
    scene,
    tracks_directory,
    x_scaling,
    y_scaling,
    z_scaling,
    napari_spline_size,
    add_surface_to_points=True,
    spline_points=100,
    fit_degree=3,
    spline_smoothing=0.05,
    point_size=30,
    spline_size=10,
    summarise_track=True,
    track_file_extension=".h5",
):

    print(
        f"Fitting splines with {spline_points} segments, of degree "
        f"'{fit_degree}' to the points"
    )
    track_files = glob(str(tracks_directory) + "/*" + track_file_extension)
    splines = []
    for track_file in track_files:
        scene, spline = analyse_track(
            scene,
            track_file,
            add_surface_to_points=add_surface_to_points,
            spline_points=spline_points,
            fit_degree=fit_degree,
            spline_smoothing=spline_smoothing,
            point_radius=point_size,
            spline_radius=spline_size,
        )
        splines.append(spline)
        if summarise_track:
            summary_csv_file = Path(track_file).with_suffix(".csv")
            analyse_track_anatomy(scene, spline, summary_csv_file)
        view_spline(
            viewer,
            base_layer,
            spline,
            x_scaling,
            y_scaling,
            z_scaling,
            napari_spline_size,
            name=Path(track_file).stem + "_fit",
        )

    return scene, splines


@thread_worker
def region_analysis(
    label_layers,
    structures_df,
    regions_directory,
    annotations_path,
    hemispheres_path,
    output_csv_file=None,
    volumes=True,
    summarise=True,
):
    if volumes:
        print("Calculating region volume distribution")
        annotations = load_any(annotations_path)
        hemispheres = load_any(hemispheres_path)

        print(f"Saving summary volumes to: {regions_directory}")
        for label_layer in label_layers:
            analyse_region_brain_areas(
                label_layer,
                regions_directory,
                annotations,
                hemispheres,
                structures_df,
            )
    if summarise:
        if output_csv_file is not None:
            print("Summarising regions")
            summarise_brain_regions(label_layers, output_csv_file)

    print("Finished!\n")


@thread_worker
def save_all(
    viewer,
    regions_directory,
    tracks_directory,
    label_layers,
    points_layers,
    image_like,
    x_scaling,
    y_scaling,
    z_scaling,
    track_file_extension=".h5",
):
    save_label_layers(regions_directory, label_layers, image_like)
    save_track_layers(
        viewer,
        tracks_directory,
        points_layers,
        x_scaling,
        y_scaling,
        z_scaling,
        track_file_extension=track_file_extension,
    )
    print("Finished!\n")


def save_label_layers(
    regions_directory, label_layers, image_like,
):
    print(f"Saving regions to: {regions_directory}")
    ensure_directory_exists(regions_directory)
    for label_layer in label_layers:
        save_regions_to_file(
            label_layer, regions_directory, image_like,
        )


def save_track_layers(
    viewer,
    tracks_directory,
    points_layers,
    x_scaling,
    y_scaling,
    z_scaling,
    track_file_extension=".h5",
):
    print(f"Saving tracks to: {tracks_directory}")
    max_z = len(viewer.layers[0].data)
    convert_and_save_points(
        points_layers,
        tracks_directory,
        x_scaling,
        y_scaling,
        z_scaling,
        max_z,
        track_file_extension=track_file_extension,
    )
