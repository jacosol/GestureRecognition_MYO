import DSformatting


path = 'path-to-dataset'
save_root = 'path-to-store-txt-raw-data'

# get all the txt files in the raw data
files = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], '*.txt'))]
# runs over all files and saves each processed sample to create the custom dataset
k = 1
for file in files[1:]:
    res = create_single_txt_sample(file, k, save_root)
    if res:
        k = k+2


raw_ds_path = 'same-as-as-save-root-above'
processed_ds_path = 'path-to-training-data'
# create folders to store training samples (data and labels) for each label
create_dataset_dirs(processed_ds_path)
print(glob.glob(os.path.join(processed_ds_path,'*')))

# loads the txts for each channel, gets the label, create stacks of 8 MYO channels and corresponding label
process_data_set(raw_ds_path, processed_ds_path)  