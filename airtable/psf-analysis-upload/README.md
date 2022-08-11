# Workflow: psf-analsyis-airtable-upload
This workflow uploads [napari-psf-analysis](https://github.com/fmi-faim/napari-psf-analysis) plugin results which are stored in a dedicated upload directory to [airtable](https://airtable.com/).
The airtable connection is only created if new data is available for upload. 

The napari-psf-analysis plugin generates a small overview png for each 
result.
Unfortunately a direct upload of `.png` files is not possible.
Hence, the `.png` files are uploaded to [cloudinary](https://cloudinary.com/) first. 
Then the download link is uploaded to airtable. 
This triggers airtable to download the image content from the remote location.
Once, the image is downloaded by airtable it is removed from the 
cloudinary storage. 
Finally, the uploaded data is locally moved to a backup storage directory.

## Parameters
* `airtable_config_path`: Config containing the airtable API information. 
* `cloudinary_config_path`: Config containing the cloudinary API information.

# Installation
We recommend installing the requirements into a fresh conda environment.
```shell
conda create -n airtable-upload python=3.9
conda activate airtable-upload
python -m pip install -r requirements.txt
```

