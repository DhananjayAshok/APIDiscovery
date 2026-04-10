source scripts/utils.sh

currdir=$(pwd)
tmp_dir=$storage_dir/tmp/coverage_files/
cd $tmp_dir
coverage erase
for file in *.py; do
    echo "Running coverage for $file"
    coverage run -p $file
done
coverage combine
coverage report -m
cd $currdir

# achieves 99% coverage on average