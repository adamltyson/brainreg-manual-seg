import napari
from brainreg_manual_seg.widgets import General


def main(
    num_colors=10,
    track_file_extension=".h5",
    image_file_extension=".tiff",
    spline_points_default=1000,
    spline_smoothing_default=0.1,
    fit_degree_default=3,
    summarise_track_default=True,
    add_surface_point_default=False,
    calculate_volumes_default=True,
    summarise_volumes_default=True,
):

    print("Loading manual segmentation GUI.\n ")
    with napari.gui_qt():

        viewer = napari.Viewer(title="Manual segmentation")
        general = General(
            viewer,
            track_file_extension=track_file_extension,
            image_file_extension=image_file_extension,
            num_colors=num_colors,
            spline_points_default=spline_points_default,
            spline_smoothing_default=spline_smoothing_default,
            fit_degree_default=fit_degree_default,
            summarise_track_default=summarise_track_default,
            add_surface_point_default=add_surface_point_default,
            calculate_volumes_default=calculate_volumes_default,
            summarise_volumes_default=summarise_volumes_default,
        )
        viewer.window.add_dock_widget(general, name="General", area="right")


if __name__ == "__main__":
    main()
