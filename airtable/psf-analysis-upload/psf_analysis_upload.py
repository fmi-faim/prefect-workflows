import configparser
import time
from glob import glob
from os.path import join, dirname, basename
from shutil import move

import cloudinary
import pandas as pd
from prefect import task, Flow, Parameter, case, unmapped
from prefect.run_configs import LocalRun
from prefect.storage import GitHub
from pyairtable import Api


@task()
def load_airtable_config(path):
    airtable_config = configparser.ConfigParser()
    airtable_config.read(path)
    return airtable_config


@task()
def get_uploaded_dir(airtable_config):
    return airtable_config['DEFAULT']['uploaded_dir']


@task()
def list_files(airtable_config):
    return glob(join(airtable_config['DEFAULT']['upload_dir'], '*.csv'))


@task()
def got_new_files(files):
    return len(files) > 0


@task()
def connect_to_table(airtable_config):
    api = Api(airtable_config['DEFAULT']['api_key'])

    return api.get_table(airtable_config['DEFAULT']['base_id'],
                         airtable_config['DEFAULT']['table_name'])


@task()
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


@task()
def load_measurement(path):
    return pd.read_csv(path)


@task()
def upload(path, data, table, uploaded_dir):
    for i in range(len(data)):
        img_name = join(dirname(path), basename(data.iloc[i]["PSF_path"]))
        import cloudinary.uploader
        response = cloudinary.uploader.upload(img_name)

        # Create table row. Handle empty comments.
        row = data.iloc[i][
            ["ImageName",
             "Date",
             "Microscope",
             "Magnification",
             "NA",
             "Amplitude",
             "Background",
             "X",
             "Y",
             "Z",
             "FWHM_X",
             "FWHM_Y",
             "FWHM_Z",
             "PrincipalAxis_1",
             "PrincipalAxis_2",
             "PrincipalAxis_3",
             "SignalToBG",
             "XYpixelsize",
             "Zspacing",
             "cov_xx",
             "cov_xy",
             "cov_xz",
             "cov_yy",
             "cov_yz",
             "cov_zz",
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
             "sde_cov_zz", ]].to_dict()

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

        row['Magnification'] = str(row['Magnification'])
        if row["Objective_id"] is not None:
            row["Objective_id"] = str(row["Objective_id"])

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


@task()
def move_uploaded(file, uploaded_dir):
    move(file, join(uploaded_dir, basename(file)))


with Flow("psf-analsyis-airtable-upload",
          run_config=LocalRun(labels=["CPU"])) as flow:
    airtable_config_path = Parameter("airtable_config_path",
                                     default="/path/to/config")
    cloudinary_config_path = Parameter("cloudinary_config_path",
                                       default="/path/to/config")

    airtable_config = load_airtable_config(airtable_config_path)
    uploaded_dir = get_uploaded_dir(airtable_config)

    files = list_files(airtable_config)

    with case(got_new_files(files), True):
        ctc = connect_to_cloudinary(cloudinary_config_path)

        table = connect_to_table(airtable_config, upstream_tasks=[
            ctc])

        data_to_upload = load_measurement.map(files)
        upload_task = upload.map(files, data_to_upload, unmapped(table),
                                 unmapped(uploaded_dir))

        move_uploaded.map(files, unmapped(uploaded_dir),
                          upstream_tasks=[upload_task])

flow.storage = GitHub(
    repo="fmi-faim/prefect-workflows",
    path="airtable/psf-analysis-upload/psf_analysis_upload.py"
)
