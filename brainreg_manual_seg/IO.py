import imio

import numpy as np
import pandas as pd

from pathlib import Path
from skimage import measure

from imlib.general.pathlib import append_to_pathlib_stem

from napari.qt.threading import thread_worker


def convert_obj_to_br(verts, faces, voxel_size):
    if voxel_size != 1:
        verts = verts * voxel_size

    faces = faces + 1
    return verts, faces


def extract_and_save_object(
    image, output_file_name, voxel_size, threshold=0, step_size=1
):
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        image, threshold, step_size=step_size
    )
    verts, faces = convert_obj_to_br(verts, faces, voxel_size)
    marching_cubes_to_obj(
        (verts, faces, normals, values), str(output_file_name)
    )


def marching_cubes_to_obj(marching_cubes_out, output_file):
    """
    Saves the output of skimage.measure.marching_cubes as an .obj file
    :param marching_cubes_out: tuple
    :param output_file: str
    """

    verts, faces, normals, _ = marching_cubes_out
    with open(output_file, "w") as f:
        for item in verts:
            f.write(f"v {item[0]} {item[1]} {item[2]}\n")
        for item in normals:
            f.write(f"vn {item[0]} {item[1]} {item[2]}\n")
        for item in faces:
            f.write(
                f"f {item[0]}//{item[0]} {item[1]}//{item[1]} "
                f"{item[2]}//{item[2]}\n"
            )
        f.close()


def volume_to_vector_array_to_obj_file(
    image,
    output_path,
    voxel_size=50,
    step_size=1,
    threshold=0,
    deal_with_regions_separately=False,
):
    # BR is oriented differently
    image = np.flip(image, axis=2)
    if deal_with_regions_separately:
        for label_id in np.unique(image):
            if label_id != 0:
                filename = append_to_pathlib_stem(
                    Path(output_path), "_" + str(label_id)
                )
                image = image == label_id
                extract_and_save_object(
                    image,
                    filename,
                    voxel_size,
                    threshold=threshold,
                    step_size=step_size,
                )
    else:
        extract_and_save_object(
            image,
            output_path,
            voxel_size,
            threshold=threshold,
            step_size=step_size,
        )


def brainrender_track_to_napari(track_file, max_z):
    points = pd.read_hdf(track_file)
    points["x"] = points["x"]
    points["z"] = points["z"]
    points["y"] = points["y"]

    points["x"] = max_z - points["x"]

    return points.to_numpy().astype(np.int16)


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


def save_regions_to_file(
    label_layer,
    destination_directory,
    ignore_empty=True,
    obj_ext=".obj",
    image_extension=".tiff",
):
    """
    Analysed the regions (to see what brain areas they are in) and saves
    the segmented regions to file (both as .obj and .nii)
    :param label_layer: napari labels layer (with segmented regions)
    :param destination_directory: Where to save files to
    :param ignore_empty: If True, don't attempt to save empty images
    :param obj_ext: File extension for the obj files
    :param image_extension: File extension fo the image files
    """
    data = label_layer.data
    if ignore_empty:
        if data.sum() == 0:
            return

    name = label_layer.name

    filename = destination_directory / (name + obj_ext)
    volume_to_vector_array_to_obj_file(
        data, filename,
    )

    filename = destination_directory / (name + image_extension)
    imio.to_tiff(data.astype(np.int16), filename)


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
