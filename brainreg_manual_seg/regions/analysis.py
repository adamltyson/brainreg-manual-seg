from napari.qt.threading import thread_worker

from brainreg_manual_seg.man_seg_tools import (
    analyse_region_brain_areas,
    summarise_brain_regions,
)


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
