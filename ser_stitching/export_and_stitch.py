import argparse
import os.path
from pathlib import Path

from prefect import Flow, Parameter, task, unmapped

from em_tasks.prefect_task.export import get_files_task, process_metadata_task
from em_tasks.prefect_task.stitch import stitch_tiles_task
from em_tasks.task.export import load_ser_file, export_uint16, export_normalized_uint8
from prefect.storage import GitHub

TILE_CONF_NAME = "TileConfiguration.txt"


@task(log_stdout=True)
def batch_export(ser_file, save_dir, intensity_range):
    print(ser_file)
    metadata, data, pixel_size = load_ser_file(ser_file=ser_file)
    basename = Path(ser_file).stem
    export_uint16(data=data, pixel_size=pixel_size, save_dir=save_dir, basename=basename)
    export_normalized_uint8(data=data, pixel_size=pixel_size, save_dir=save_dir, basename=basename,
                            intensity_range=intensity_range)
    metadata['image_file_name'] = basename + '.tif'
    metadata['pixel_size'] = pixel_size[0] * 1000 * 1000
    return metadata


@task()
def path_join_task(folder, file):
    return os.path.join(folder, file)


flow: Flow
with Flow("ser-stitching") as flow:
    input_dir = Parameter("input_dir", default=r'/path/to/input/dir')
    filename_filter = Parameter("filename_filter", default="*.ser")
    save_dir = Parameter("save_dir", default=r'/path/to/output/dir')
    intensity_range = Parameter("intensity_range", default=1000)
    files = get_files_task(input_dir=input_dir, filename_filter=filename_filter)
    metadata = batch_export.map(ser_file=files, save_dir=unmapped(save_dir), intensity_range=unmapped(intensity_range))
    # alternative: directly work with mapped results
    # TODO doesn't seem to work in prefect 1.x, see https://github.com/PrefectHQ/prefect/issues/4839
    # metadata, data, pixel_size = load_ser_file_task.map(ser_file=files)
    # basenames = [os.path.basename(f) for f in files]  # TODO create task for this?
    # export_uint16_task.map(data=data, pixel_size=pixel_size, save_dir=unmapped(save_dir), basename=basenames)
    # export_normalized_uint8_task.map(data=data, pixel_size=pixel_size, save_dir=unmapped(save_dir),
    #                                  basename=basenames, intensity_range=intensity_range)
    # save_position_file from metadata
    process_metadata = process_metadata_task(metadata_list=metadata, save_dir=save_dir, prefixes=["8bit", "16bit"],
                          filename=TILE_CONF_NAME)
    # stitch tiles
    subdir = path_join_task(save_dir, "8bit")
    stitch_tiles_task(input_dir=subdir, tileconf_filename=TILE_CONF_NAME,
                      upstream_tasks=[process_metadata])

flow.storage = GitHub(
    repo="fmi-faim/prefect-workflows",
    path="ser_stitching/export_and_stitch.py",
    ref="dev-ser-export",
    access_token_secret="github-access-token_buchtimo"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    flow.run(input_dir=args.input_dir, save_dir=args.save_dir)
