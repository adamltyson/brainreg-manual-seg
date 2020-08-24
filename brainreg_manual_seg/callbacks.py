import numpy as np
from glob import glob
from pathlib import Path
from napari.qt.threading import thread_worker
from scipy.spatial import cKDTree

from brainreg_manual_seg.man_seg_tools import (
    save_regions_to_file,
    analyse_region_brain_areas,
    summarise_brain_regions,
    analyse_track,
    analyse_track_anatomy,
    convert_and_save_points,
)


def track_analysis(
    viewer,
    atlas,
    tracks_directory,
    track_layers,
    napari_spline_size,
    spline_points=100,
    fit_degree=3,
    spline_smoothing=0.05,
    summarise_track=True,
):
    tracks_directory.mkdir(parents=True, exist_ok=True)

    print(
        f"Fitting splines with {spline_points} segments, of degree "
        f"'{fit_degree}' to the points"
    )
    splines = []

    for track_layer in track_layers:
        spline = analyse_track(
            track_layer,
            spline_points=spline_points,
            fit_degree=fit_degree,
            spline_smoothing=spline_smoothing,
        )
        splines.append(spline)
        if summarise_track:
            summary_csv_file = tracks_directory / (track_layer.name + ".csv")
            analyse_track_anatomy(atlas, spline, summary_csv_file)

        viewer.add_points(
            spline,
            size=napari_spline_size,
            edge_color="cyan",
            face_color="cyan",
            blending="additive",
            opacity=0.7,
            name=track_layer.name + "_fit",
        )

    return splines


@thread_worker
def region_analysis(
    label_layers,
    atlas_layer_image,
    atlas,
    regions_directory,
    output_csv_file=None,
    volumes=True,
    summarise=True,
):
    regions_directory.mkdir(parents=True, exist_ok=True)
    if volumes:
        print("Calculating region volume distribution")
        print(f"Saving summary volumes to: {regions_directory}")
        for label_layer in label_layers:
            analyse_region_brain_areas(
                label_layer, atlas_layer_image, regions_directory, atlas,
            )
    if summarise:
        if output_csv_file is not None:
            print("Summarising regions")
            summarise_brain_regions(
                label_layers, output_csv_file, atlas.resolution
            )

    print("Finished!\n")


@thread_worker
def save_all(
    regions_directory,
    tracks_directory,
    label_layers,
    points_layers,
    track_file_extension=".h5",
):
    if label_layers:
        save_label_layers(regions_directory, label_layers)

    if points_layers:
        save_track_layers(
            tracks_directory,
            points_layers,
            track_file_extension=track_file_extension,
        )
    print("Finished!\n")


def save_label_layers(regions_directory, label_layers):
    print(f"Saving regions to: {regions_directory}")
    regions_directory.mkdir(parents=True, exist_ok=True)
    for label_layer in label_layers:
        save_regions_to_file(label_layer, regions_directory)


def save_track_layers(
    tracks_directory, points_layers, track_file_extension=".h5",
):
    print(f"Saving tracks to: {tracks_directory}")
    convert_and_save_points(
        points_layers,
        tracks_directory,
        track_file_extension=track_file_extension,
    )
