# Create Prefect Deployment

## shading_reference_yokogawa.py
`prefect deployment build eicm_flows/shading_reference_yokogawa.py:create_shading_reference_yokogawa -n "default" -q slurm -sb github/prefect-workflows-eicm --skip-upload -o eicm_flows/deployment/create_shading_reference_yokogawa.yaml -ib process/slurm-prefect-workflows-eicm -t fiji -t eicm`

## fit_gaussian_estimation.py
`prefect deployment build eicm_flows/fit_gaussian_estimation.py:eicm_gaussian_fit -n "default" -q slurm -sb github/prefect-workflows-eicm --skip-upload -o eicm_flows/deployment/fit_gaussian_estimation.yaml -ib process/slurm-prefect-workflows-eicm -t fiji -t eicm`

## fit_polynomial_estimation.py
`prefect deployment build eicm_flows/fit_polynomial_estimation.py:eicm_polynomial_fit -n "default" -q slurm -sb github/prefect-workflows-eicm --skip-upload -o eicm_flows/deployment/fit_polynomial_estimation.yaml -ib process/slurm-prefect-workflows-eicm -t fiji -t eicm`

## median_filter_estimation.py
`prefect deployment build eicm_flows/median_filter_estimation.py:eicm_median_filter -n "default" -q slurm -sb github/prefect-workflows-eicm --skip-upload -o eicm_flows/deployment/median_filter_estimation.yaml -ib process/slurm-prefect-workflows-eicm -t fiji -t eicm`

## compute_all_matrices_yokogawa.py
`prefect deployment build eicm_flows/compute_all_matrices.py:eicm_all_yokogawa -n "default" -q slurm -sb github/prefect-workflows-eicm --skip-upload -o eicm_flows/deployment/compute_all_matrices_yokogawa.yaml -ib process/slurm-prefect-workflows-eicm -t fiji -t eicm`

## run_all_estimations.py
`prefect deployment build eicm_flows/run_all_estimations.py:eicm_all -n "default" -q slurm -sb github/prefect-workflows-eicm --skip-upload -o eicm_flows/deployment/run_all_estimations.yaml -ib process/slurm-prefect-workflows-eicm -t fiji -t eicm`

## Apply
`prefect deployment apply eicm_flows/deployment/*.yaml`