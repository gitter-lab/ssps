#! /bin/bash
#
# A simple test that runs SSPS and makes sure it returns a 
# reasonable-looking CSV file. It compares the SSPS output 
# against `baseline_predictions.csv`.

exit_status=0

cd ../run_ssps
snakemake --cores 2
csvdiff --style=summary --sep=',' --ignore-columns=score --output=csvdiff.out node1,node2 example_predictions.csv ../tests/baseline_predictions.csv

comparison=$(cat csvdiff.out)

echo $comparison

rm csvdiff.out
if [[ "$comparison" != 'files are identical' ]]; then
    exit_status=1
fi

exit $exit_status
