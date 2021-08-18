import pandas as pd
from shutil import copyfile

# load cleaned & merged hemi data
lookup_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\hemi_lookup_cleaned.csv"
date = "19_149"
src_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\" + date + "\\"
dst_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\" + date + "\\clean\\"

# load lookup
lookup = pd.read_csv(lookup_in)
# filter for quality 4 or lower
lookup = lookup.loc[lookup.quality_code <= 4]
# filter for date
lookup = lookup.loc[lookup.folder == date]

# copy images to separate folder for processing
for ii in range(0, lookup.shape[0]):
    src_path = src_dir + lookup.filename.iloc[ii]
    dst_path = dst_dir + lookup.filename.iloc[ii]
    copyfile(src_path, dst_path)

# process clean folder in hemisfer