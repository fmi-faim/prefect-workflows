
# Build deployment
`prefect deployment build airtable/psf-analysis-upload/psf_analysis_upload.py:psf_analysis_airtable_upload -n "default" -q slurm -sb github/prefect-workflows-airtable-upload --skip-upload -o airtable/psf-analysis-upload/deployment/psf_analysis_upload.yaml -ib process/slurm-prefect-workflows-airtable-upload -t faim -t log`

# Apply depoloyment
`prefect deployment apply airtable/psf-analysis-upload/deployment/*.yaml`