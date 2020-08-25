import numpy as np
import pandas as pd


def save_track_layers(
    tracks_directory, points_layers, track_file_extension=".h5",
):
    print(f"Saving tracks to: {tracks_directory}")
    convert_and_save_points(
        points_layers,
        tracks_directory,
        track_file_extension=track_file_extension,
    )


def convert_and_save_points(
    points_layers, output_directory, track_file_extension=".h5",
):
    """
    Converts the points from the napari format (in image space) to brainrender
    (in atlas space)
    :param points_layers: list of points layers
    :param output_directory: path to save points to
    """

    output_directory.mkdir(parents=True, exist_ok=True)

    for points_layer in points_layers:
        save_single_track_layer(
            points_layer,
            output_directory,
            track_file_extension=track_file_extension,
        )


def save_single_track_layer(
    layer, output_directory, track_file_extension=".h5",
):
    output_filename = output_directory / (layer.name + track_file_extension)
    cells = layer.data.astype(np.int16)
    cells = pd.DataFrame(cells)

    cells.columns = ["x", "y", "z"]
    cells.to_hdf(output_filename, key="df", mode="w")


def brainrender_track_to_napari(track_file, max_z):
    points = pd.read_hdf(track_file)
    points["x"] = points["x"]
    points["z"] = points["z"]
    points["y"] = points["y"]

    points["x"] = max_z - points["x"]

    return points.to_numpy().astype(np.int16)
