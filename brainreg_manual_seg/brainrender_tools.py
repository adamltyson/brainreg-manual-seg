from glob import glob

from brainrender.scene import Scene
from imlib.plotting.colors import get_random_vtkplotter_color


def load_obj_into_brainrender(
    scene, obj_file, color=None, alpha=0.8, shading="phong"
):
    """
    Loads a single obj file into brainrender
    :param scene: brainrender scene
    :param obj_file: obj filepath
    :param color: Object color. If None, a random color is chosen
    :param alpha: Object transparency
    :param shading: Object shading type ("flat", "giroud" or "phong").
    Defaults to "phong"
    """
    obj_file = str(obj_file)
    if color is None:
        color = get_random_vtkplotter_color()
    act = scene.add_from_file(obj_file, c=color, alpha=alpha)

    if shading == "flat":
        act.GetProperty().SetInterpolationToFlat()
    elif shading == "gouraud":
        act.GetProperty().SetInterpolationToGouraud()
    else:
        act.GetProperty().SetInterpolationToPhong()


def load_regions_into_brainrender(
    scene, list_of_regions, alpha=0.8, shading="flat"
):
    """
    Loads a list of .obj files into brainrender
    :param scene: brainrender scene
    :param list_of_regions: List of .obj files to be loaded
    :param alpha: Object transparency
    :param shading: Object shading type ("flat", "giroud" or "phong").
    Defaults to "phong"
    """
    for obj_file in list_of_regions:
        load_obj_into_brainrender(
            scene, obj_file, alpha=alpha, shading=shading
        )
    return scene


def display_track_in_brainrender(scene, spline, verbose=True):
    """

    :param scene: brainrender scene object
    :param spline: vtkplotter spline object
    :param bool verbose: Whether to print the progress
    """
    if verbose:
        print("Visualising 3D data in brainrender")

    scene.add_vtkactor(spline)
    scene.verbose = False
    return scene


def view_in_brainrender(
    spline,
    regions_directory,
    alpha=0.8,
    shading="flat",
    regions_to_add=[],
    region_alpha=0.3,
):
    scene = Scene(add_root=True)
    obj_files = glob(str(regions_directory) + "/*.obj")
    scene.add_brain_regions(regions_to_add, alpha=region_alpha)

    if obj_files:
        scene = load_regions_into_brainrender(
            scene, obj_files, alpha=alpha, shading=shading
        )
    try:
        scene = display_track_in_brainrender(scene, spline,)
    except:
        pass

    scene.render()
