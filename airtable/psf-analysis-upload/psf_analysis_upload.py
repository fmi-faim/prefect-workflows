import configparser
import time
from glob import glob
from os.path import join, dirname, basename
from shutil import move
from typing import List, Dict

import cloudinary
import pandas as pd
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from pyairtable import Api


def load_airtable_config(path):
    airtable_config = configparser.ConfigParser()
    airtable_config.read(path)
    return airtable_config


def list_files(airtable_config):
    return glob(join(airtable_config['DEFAULT']['upload_dir'], '*.csv'))


def connect_to_table(airtable_config):
    api = Api(airtable_config['DEFAULT']['api_key'])

    return api.table(airtable_config['DEFAULT']['base_id'],
                         airtable_config['DEFAULT']['table_name'])


def connect_to_cloudinary(path):
    cloudinary_config = configparser.ConfigParser()
    cloudinary_config.read(path)

    # Set cloudinary config before any cloudinary imports.
    # Necessary to get the api_proxy working.
    cloudinary.config(
        cloud_name=cloudinary_config['DEFAULT']['cloud_name'],
        api_key=cloudinary_config['DEFAULT']['api_key'],
        api_secret=cloudinary_config['DEFAULT']['api_secret'],
        secure=True,
        api_proxy=cloudinary_config['DEFAULT']['api_proxy']
    )


def rename_columns(row: dict):
    renamed = {
        "ImageName": row["ImageName"],
        "Date": row["Date"],
        "Microscope": row["Microscope"],
        "Magnification": row["Magnification"],
        "NA": row["NA"],
        "Amplitude_3D_XYZ": row["Amplitude"],
        "Amplitude_2D_XY": row["Amplitude_2D"],
        "Background_3D_XYZ": row["Background"],
        "Background_2D_XY": row["Background_2D"],
        "X_3D": row["X"],
        "Y_3D": row["Y"],
        "Z_3D": row["Z"],
        "X_2D": row["X_2D"],
        "Y_2D": row["Y_2D"],
        "FWHM_3D_X": row["FWHM_X"],
        "FWHM_3D_Y": row["FWHM_Y"],
        "FWHM_3D_Z": row["FWHM_Z"],
        "FWHM_2D_X": row["FWHM_X_2D"],
        "FWHM_2D_Y": row["FWHM_Y_2D"],
        "FWHM_PA1_3D": row["PrincipalAxis_1"],
        "FWHM_PA2_3D": row["PrincipalAxis_2"],
        "FWHM_PA3_3D": row["PrincipalAxis_3"],
        "FWHM_PA1_2D": row["PrincipalAxis_1_2D"],
        "FWHM_PA2_2D": row["PrincipalAxis_2_2D"],
        "SignalToBG_3D_XYZ": row["SignalToBG"],
        "SignalToBG_2D_XY": row["SignalToBG_2D"],
        "XYpixelsize": row["XYpixelsize"],
        "Zspacing": row["Zspacing"],
        "cov_xx_3D": row["cov_xx"],
        "cov_xy_3D": row["cov_xy"],
        "cov_xz_3D": row["cov_xz"],
        "cov_yy_3D": row["cov_yy"],
        "cov_yz_3D": row["cov_yz"],
        "cov_zz_3D": row["cov_zz"],
        "cov_xx_2D": row["cov_xx_2D"],
        "cov_xy_2D": row["cov_xy_2D"],
        "cov_yy_2D": row["cov_yy_2D"],
        "sde_amp_3D_XYZ": row["sde_peak"],
        "sde_background_3D_XYZ": row["sde_background"],
        "sde_X_3D": row["sde_X"],
        "sde_Y_3D": row["sde_Y"],
        "sde_Z_3D": row["sde_Z"],
        "sde_cov_xx_3D": row["sde_cov_xx"],
        "sde_cov_xy_3D": row["sde_cov_xy"],
        "sde_cov_xz_3D": row["sde_cov_xz"],
        "sde_cov_yy_3D": row["sde_cov_yy"],
        "sde_cov_yz_3D": row["sde_cov_yz"],
        "sde_cov_zz_3D": row["sde_cov_zz"],
        "sde_amp_2D_XY": row["sde_peak_2D"],
        "sde_background_2D_XY": row["sde_background_2D"],
        "sde_X_2D": row["sde_X_2D"],
        "sde_Y_2D": row["sde_Y_2D"],
        "sde_cov_xx_2D": row["sde_cov_xx_2D"],
        "sde_cov_xy_2D": row["sde_cov_xy_2D"],
        "sde_cov_yy_2D": row["sde_cov_yy_2D"],
        "version": row["version"],
    }
    return renamed



def upload(path, table, uploaded_dir):
    data = pd.read_csv(path)
    for i in range(len(data)):
        img_name = join(dirname(path), basename(data.iloc[i]["PSF_path"]))
        import cloudinary.uploader
        response = cloudinary.uploader.upload(img_name)

        version = data.iloc[i]["version"]

        if version.startswith("0."):
            row = create_row_v0(data, i)
            row = rename_columns(row)
        else:
            row = create_row_v1(data, i)

        # Provide the url of the PSF image.
        # Airtable will fetch the image from there.
        # Direct image upload is not supported by the Airtable API.
        row['PSF_Image'] = [{'url': response['secure_url']}]

        # Create a new entry in the Airtable table.
        row_id = table.create(row)['id']

        # Probing if the thumbnail has been created.
        # If the thumbnail is there, it means that Airtable has downloaded the
        # image from cloudinary.
        rec = table.get(row_id)
        while not 'thumbnails' in rec['fields']['PSF_Image'][0].keys():
            time.sleep(1)
            rec = table.get(row_id)

        # Delete the image from cloudinary.
        cloudinary.uploader.destroy(response['public_id'])

        # Move the uploaded image to the uploaded directory.
        move(img_name, join(uploaded_dir,
                            basename(img_name)))


def create_row_v0(data, i):
    # Create table row. Handle empty comments.
    row = data.iloc[i][
        [
            "ImageName",
            "Date",
            "Microscope",
            "Magnification",
            "NA",
            "Amplitude",
            "Amplitude_2D",
            "Background",
            "Background_2D",
            "X",
            "Y",
            "Z",
            "X_2D",
            "Y_2D",
            "FWHM_X",
            "FWHM_Y",
            "FWHM_Z",
            "FWHM_X_2D",
            "FWHM_Y_2D",
            "PrincipalAxis_1",
            "PrincipalAxis_2",
            "PrincipalAxis_3",
            "PrincipalAxis_1_2D",
            "PrincipalAxis_2_2D",
            "SignalToBG",
            "SignalToBG_2D",
            "XYpixelsize",
            "Zspacing",
            "cov_xx",
            "cov_xy",
            "cov_xz",
            "cov_yy",
            "cov_yz",
            "cov_zz",
            "cov_xx_2D",
            "cov_xy_2D",
            "cov_yy_2D",
            "sde_peak",
            "sde_background",
            "sde_X",
            "sde_Y",
            "sde_Z",
            "sde_cov_xx",
            "sde_cov_xy",
            "sde_cov_xz",
            "sde_cov_yy",
            "sde_cov_yz",
            "sde_cov_zz",
            "sde_peak_2D",
            "sde_background_2D",
            "sde_X_2D",
            "sde_Y_2D",
            "sde_cov_xx_2D",
            "sde_cov_xy_2D",
            "sde_cov_yy_2D",
            "version",
        ]].to_dict()

    def add_field(name, r, cast):
        if name in data.columns:
            r[name] = cast(data.iloc[i][name])
        else:
            r[name] = None

    add_field("sde_fwhm_x", row, float)
    add_field("sde_fwhm_y", row, float)
    add_field("sde_fwhm_z", row, float)
    add_field("Objective_id", row, str)
    add_field("Temperature", row, int)
    add_field("AiryUnit", row, int)
    add_field("BeadSize", row, int)
    add_field("BeadSupplier", row, str)
    add_field("MountingMedium", row, str)
    add_field("Operator", row, str)
    add_field("MicroscopeType", row, str)
    add_field("Excitation", row, int)
    add_field("Emission", row, int)
    add_field("Comment", row, str)
    add_field("End date", row, str)
    row['Magnification'] = str(row['Magnification'])
    if row["Objective_id"] is not None:
        row["Objective_id"] = str(row["Objective_id"])
    return row


def create_row_v1(data, i):
    # Create table row. Handle empty comments.
    row = data.iloc[i][
        [
            "ImageName",
            "Date",
            "Microscope",
            "Magnification",
            "NA",
            "Amplitude_1D_Z",
            "Amplitude_2D_XY",
            "Amplitude_3D_XYZ",
            "Background_1D_Z",
            "Background_2D_XY",
            "Background_3D_XYZ",
            "Z_1D",
            "X_2D",
            "Y_2D",
            "X_3D",
            "Y_3D",
            "Z_3D",
            "FWHM_1D_Z",
            "FWHM_2D_X",
            "FWHM_2D_Y",
            "FWHM_3D_Z",
            "FWHM_3D_Y",
            "FWHM_3D_X",
            "FWHM_PA1_2D",
            "FWHM_PA2_2D",
            "FWHM_PA1_3D",
            "FWHM_PA2_3D",
            "FWHM_PA3_3D",
            "SignalToBG_1D_Z",
            "SignalToBG_2D_XY",
            "SignalToBG_3D_XYZ",
            "XYpixelsize",
            "Zspacing",
            "cov_xx_3D",
            "cov_xy_3D",
            "cov_xz_3D",
            "cov_yy_3D",
            "cov_yz_3D",
            "cov_zz_3D",
            "cov_xx_2D",
            "cov_xy_2D",
            "cov_yy_2D",
            "sde_amp_1D_Z",
            "sde_amp_2D_XY",
            "sde_amp_3D_XYZ",
            "sde_background_1D_Z",
            "sde_background_2D_XY",
            "sde_background_3D_XYZ",
            "sde_Z_1D",
            "sde_X_2D",
            "sde_Y_2D",
            "sde_X_3D",
            "sde_Y_3D",
            "sde_Z_3D",
            "sde_cov_xx_3D",
            "sde_cov_xy_3D",
            "sde_cov_xz_3D",
            "sde_cov_yy_3D",
            "sde_cov_yz_3D",
            "sde_cov_zz_3D",
            "sde_cov_xx_2D",
            "sde_cov_xy_2D",
            "sde_cov_yy_2D",
            "version",
        ]].to_dict()

    def add_field(name, r, cast):
        if name in data.columns:
            r[name] = cast(data.iloc[i][name])
        else:
            r[name] = None

    add_field("Objective_id", row, str)
    add_field("Temperature", row, int)
    add_field("AiryUnit", row, int)
    add_field("BeadSize", row, int)
    add_field("BeadSupplier", row, str)
    add_field("MountingMedium", row, str)
    add_field("Operator", row, str)
    add_field("MicroscopeType", row, str)
    add_field("Excitation", row, int)
    add_field("Emission", row, int)
    add_field("Comment", row, str)
    add_field("End date", row, str)
    row['Magnification'] = str(row['Magnification'])
    if row["Objective_id"] is not None:
        row["Objective_id"] = str(row["Objective_id"])
    return row


@task()
def upload_and_move(files: List[str],
                    cloudinary_config_path: str,
                    airtable_config: Dict):

    uploaded_dir = airtable_config['DEFAULT']['uploaded_dir']

    connect_to_cloudinary(cloudinary_config_path)
    table = connect_to_table(airtable_config)

    for file in files:
        upload(file, table, uploaded_dir)
        move(file, join(uploaded_dir, basename(file)))


@flow(
    name="PSF Analysis Airtable Upload",
    task_runner=SequentialTaskRunner()
)
def psf_analysis_airtable_upload(
        airtable_config_path: str = "/path/to/config",
        cloudinary_config_path: str = "/path/to/config",
):
    airtable_config = load_airtable_config(airtable_config_path)

    files = list_files(airtable_config)

    if len(files) > 0:
        upload_and_move.submit(files, cloudinary_config_path, airtable_config)
