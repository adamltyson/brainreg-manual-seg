import imio

import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path
from skimage.measure import regionprops_table
from vedo import mesh, Spheres, Spline

from imlib.pandas.misc import initialise_df
from imlib.general.list import unique_elements_lists
from imlib.general.pathlib import append_to_pathlib_stem

from skimage import measure


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


def lateralise_atlas(
    atlas, hemispheres, left_hemisphere_value=1, right_hemisphere_value=2
):
    atlas_left = atlas[hemispheres == left_hemisphere_value]
    atlas_right = atlas[hemispheres == right_hemisphere_value]
    return atlas_left, atlas_right


def add_new_label_layer(
    viewer,
    base_image,
    name="region",
    selected_label=1,
    num_colors=10,
    brush_size=30,
):
    """
    Takes an existing napari viewer, and adds a blank label layer
    (same shape as base_image)
    :param viewer: Napari viewer instance
    :param np.array base_image: Underlying image (for the labels to be
    referencing)
    :param str name: Name of the new labels layer
    :param int selected_label: Label ID to be preselected
    :param int num_colors: How many colors (labels)
    :param int brush_size: Default size of the label brush
    :return label_layer: napari labels layer
    """
    labels = np.empty_like(base_image)
    label_layer = viewer.add_labels(labels, num_colors=num_colors, name=name)
    label_layer.selected_label = selected_label
    label_layer.brush_size = brush_size
    return label_layer


def summarise_brain_regions(label_layers, filename, atlas_resolution):
    summaries = []
    for label_layer in label_layers:
        summaries.append(summarise_single_brain_region(label_layer))

    result = pd.concat(summaries)
    # TODO: use atlas.space to make these more intuitive
    volume_header = "volume_mm3"
    length_columns = [
        "axis_0_min_um",
        "axis_1_min_um",
        "axis_2_min_um",
        "axis_0_max_um",
        "axis_1_max_um",
        "axis_2_max_um",
        "axis_0_center_um",
        "axis_1_center_um",
        "axis_2_center_um",
    ]

    result.columns = ["region"] + [volume_header] + length_columns

    voxel_volume_in_mm = np.prod(atlas_resolution) / (1000 ** 3)

    result[volume_header] = result[volume_header] * voxel_volume_in_mm

    for header in length_columns:
        for dim, idx in enumerate(atlas_resolution):
            if header.startswith(f"axis_{idx}"):
                scale = float(dim)
                assert scale > 0
                result[header] = result[header] * scale

    result.to_csv(filename, index=False)


def summarise_single_brain_region(
    label_layer,
    ignore_empty=True,
    properties_to_fetch=["area", "bbox", "centroid",],
):
    data = label_layer.data
    if ignore_empty:
        if data.sum() == 0:
            return

    regions_table = regionprops_table(data, properties=properties_to_fetch)
    df = pd.DataFrame.from_dict(regions_table)
    df.insert(0, "Region", label_layer.name)
    return df


def add_existing_track_layers(viewer, track_file, point_size):
    max_z = len(viewer.layers[0].data)
    data = brainrender_track_to_napari(track_file, max_z)
    new_points_layer = viewer.add_points(
        data, n_dimensional=True, size=point_size, name=Path(track_file).stem,
    )
    new_points_layer.mode = "ADD"
    return new_points_layer


def brainrender_track_to_napari(track_file, max_z):
    points = pd.read_hdf(track_file)
    points["x"] = points["x"]
    points["z"] = points["z"]
    points["y"] = points["y"]

    points["x"] = max_z - points["x"]

    return points.to_numpy().astype(np.int16)


def add_existing_label_layers(
    viewer, label_file, selected_label=1, num_colors=10, brush_size=30,
):
    """
    Loads an existing image as a napari labels layer
    :param viewer: Napari viewer instance
    :param label_file: Filename of the image to be loaded
    :param int selected_label: Label ID to be preselected
    :param int num_colors: How many colors (labels)
    :param int brush_size: Default size of the label brush
    :return label_layer: napari labels layer
    """
    label_file = Path(label_file)
    labels = imio.load_any(label_file)
    label_layer = viewer.add_labels(
        labels, num_colors=num_colors, name=label_file.stem
    )
    label_layer.selected_label = selected_label
    label_layer.brush_size = brush_size
    return label_layer


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


def analyse_region_brain_areas(
    label_layer,
    atlas_layer_data,
    destination_directory,
    atlas,
    extension=".csv",
    ignore_empty=True,
):
    """

    :param label_layer: napari labels layer (with segmented regions)

    :param ignore_empty: If True, don't analyse empty regions
    """

    data = label_layer.data
    if ignore_empty:
        if data.sum() == 0:
            return

    name = label_layer.name

    masked_annotations = data.astype(bool) * atlas_layer_data

    annotations_left, annotations_right = lateralise_atlas(
        masked_annotations,
        atlas.hemispheres,
        left_hemisphere_value=atlas.left_hemisphere_value,
        right_hemisphere_value=atlas.right_hemisphere_value,
    )

    unique_vals_left, counts_left = np.unique(
        annotations_left, return_counts=True
    )
    unique_vals_right, counts_right = np.unique(
        annotations_right, return_counts=True
    )
    voxel_volume_in_mm = np.prod(atlas.resolution) / (1000 ** 3)

    df = initialise_df(
        "structure_name",
        "left_volume_mm3",
        "left_percentage_of_total",
        "right_volume_mm3",
        "right_percentage_of_total",
        "total_volume_mm3",
        "percentage_of_total",
    )

    sampled_structures = unique_elements_lists(
        list(unique_vals_left) + list(unique_vals_right)
    )
    total_volume_region = get_total_volume_regions(
        unique_vals_left, unique_vals_right, counts_left, counts_right
    )

    for atlas_value in sampled_structures:
        if atlas_value != 0:
            try:
                df = add_structure_volume_to_df(
                    df,
                    atlas_value,
                    atlas.structures,
                    unique_vals_left,
                    unique_vals_right,
                    counts_left,
                    counts_right,
                    voxel_volume_in_mm,
                    total_volume_voxels=total_volume_region,
                )

            except KeyError:
                print(
                    f"Value: {atlas_value} is not in the atlas structure"
                    f" reference file. Not calculating the volume"
                )
    filename = destination_directory / (name + extension)
    df.to_csv(filename, index=False)


def get_total_volume_regions(
    unique_vals_left, unique_vals_right, counts_left, counts_right,
):
    zero_index_left = np.where(unique_vals_left == 0)[0][0]
    counts_left = list(counts_left)
    counts_left.pop(zero_index_left)

    zero_index_right = np.where(unique_vals_right == 0)[0][0]
    counts_right = list(counts_right)
    counts_right.pop(zero_index_right)

    return sum(counts_left + counts_right)


def add_structure_volume_to_df(
    df,
    atlas_value,
    atlas_structures,
    unique_vals_left,
    unique_vals_right,
    counts_left,
    counts_right,
    voxel_volume,
    total_volume_voxels=None,
):
    name = atlas_structures[atlas_value]["name"]

    left_volume, left_percentage = get_volume_in_hemisphere(
        atlas_value,
        unique_vals_left,
        counts_left,
        total_volume_voxels,
        voxel_volume,
    )
    right_volume, right_percentage = get_volume_in_hemisphere(
        atlas_value,
        unique_vals_right,
        counts_right,
        total_volume_voxels,
        voxel_volume,
    )
    if total_volume_voxels is not None:
        total_percentage = left_percentage + right_percentage
    else:
        total_percentage = 0

    df = df.append(
        {
            "structure_name": name,
            "left_volume_mm3": left_volume,
            "left_percentage_of_total": left_percentage,
            "right_volume_mm3": right_volume,
            "right_percentage_of_total": right_percentage,
            "total_volume_mm3": left_volume + right_volume,
            "percentage_of_total": total_percentage,
        },
        ignore_index=True,
    )
    return df


def get_volume_in_hemisphere(
    atlas_value, unique_vals, counts, total_volume_voxels, voxel_volume
):
    try:
        index = np.where(unique_vals == atlas_value)[0][0]
        volume = counts[index] * voxel_volume
        if total_volume_voxels is not None:
            percentage = 100 * (counts[index] / total_volume_voxels)
        else:
            percentage = 0
    except IndexError:
        volume = 0
        percentage = 0

    return volume, percentage


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


def analyse_track(
    scene,
    track_layer,
    spline_points=100,
    fit_degree=3,
    spline_smoothing=0.05,
    point_radius=30,
    spline_radius=10,
):
    """
    Given a file of points, fit a spline function, and add to a brainrender
     scene.
    :param scene: brainrender scene object
    :param track_layer: napari points layer
    :param spline_points: How many points define the spline
    :param fit_degree: spline fit degree
    :param spline_smoothing: spline fit smoothing
    :param point_radius: size of the points in the brainrender scene
    :param spline_radius: size of the rendered spline in the brainrender
    scene
    :return:
        scene: brainrender scene with the surface point added.
        spline: vedo spline object
    """

    points = track_layer.data.astype(np.int16)
    points = pd.DataFrame(points)

    points.columns = ["x", "y", "z"]
    scene.add_cells(
        points,
        color_by_region=True,
        res=12,
        radius=point_radius,
        verbose=False,
    )
    points = np.array(points)

    far_point = np.expand_dims(points[-1], axis=0)
    scene.add_actor(Spheres(far_point, r=point_radius).color("n"))

    spline = (
        Spline(
            points,
            smooth=spline_smoothing,
            degree=fit_degree,
            res=spline_points,
        )
        .pointSize(spline_radius)
        .color("n")
    )

    return scene, spline


def analyse_track_anatomy(atlas, spline, file_path, verbose=True):
    """
    For a given spline, and brainrender scene, find the brain region that each
    "segment" is in, and save to csv.

    :param scene: brainrender scene object
    :param spline: vtkplotter spline object
    :param file_path: path to save the results to
    :param bool verbose: Whether to print the progress
    """
    if verbose:
        print("Determining the brain region for each segment of the spline")
    spline_regions = []
    for p in spline.points().tolist():
        try:
            spline_regions.append(
                atlas.structures[atlas.structure_from_coords(p)]
            )
        except KeyError:
            spline_regions.append(None)

    df = pd.DataFrame(
        columns=["Position", "Region ID", "Region acronym", "Region name"]
    )
    for idx, spline_region in enumerate(spline_regions):
        if spline_region is None:
            df = df.append(
                {
                    "Position": idx,
                    "Region ID": "Not found in brain",
                    "Region acronym": "Not found in brain",
                    "Region name": "Not found in brain",
                },
                ignore_index=True,
            )
        else:
            df = df.append(
                {
                    "Position": idx,
                    "Region ID": spline_region["id"],
                    "Region acronym": spline_region["acronym"],
                    "Region name": spline_region["name"],
                },
                ignore_index=True,
            )
    if verbose:
        print(f"Saving results to: {file_path}")
    df.to_csv(file_path, index=False)


def add_new_track_layer(viewer, track_layers, point_size):
    num = len(track_layers)
    new_track_layers = viewer.add_points(
        n_dimensional=True, size=point_size, name=f"track_{num}",
    )
    new_track_layers.mode = "ADD"
    track_layers.append(new_track_layers)


def add_new_region_layer(
    viewer, label_layers, image_like, brush_size, num_colors
):
    num = len(label_layers)
    new_label_layer = add_new_label_layer(
        viewer,
        image_like,
        name=f"region_{num}",
        brush_size=brush_size,
        num_colors=num_colors,
    )
    new_label_layer.mode = "PAINT"
    label_layers.append(new_label_layer)


def add_existing_region_segmentation(
    directory, viewer, label_layers, file_extension
):
    label_files = glob(str(directory) + "/*" + file_extension)
    if directory and label_files != []:
        for label_file in label_files:
            label_layers.append(add_existing_label_layers(viewer, label_file))
