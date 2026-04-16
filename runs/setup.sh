source scripts/utils.sh



bash runs/stats.sh

cp $storage_dir/sync/data/parquets/* $storage_dir/data/parquets/
cp $storage_dir/sync/data/finetuning/* $storage_dir/data/finetuning/
cp $storage_dir/sync/data/csvs/* $storage_dir/data/csvs/
cp $storage_dir/sync/data/final/* $storage_dir/data/final/

