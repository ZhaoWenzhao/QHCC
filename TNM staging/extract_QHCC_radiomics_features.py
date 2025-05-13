## @package extract_QHCC_radiomics_features
# @version v1.0
# @brief Extracting radiomics features from medical images with given masks
from radiomics import featureextractor
import os
import pandas as pd
import SimpleITK as sitk

## Initialize the PyRadiomics feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

## File paths: MRI image and Mask file directories
mri_dir = "./data/L1_files"
mask_dir = "./data/L1_files_mask"

## Get lists of MRI and Mask files
# default file format: ".nii"
mri_files = sorted([f for f in os.listdir(mri_dir) if f.endswith(".nii")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii")])

## Ensure the number of MRI and Mask files match
if len(mri_files) != len(mask_files):
    raise ValueError(f"Number of MRI files ({len(mri_files)}) does not match number of Mask files ({len(mask_files)})")

## Initialize a DataFrame to store the features
radiomics_data = pd.DataFrame()

## Batch processing
for mri_file, mask_file in zip(mri_files, mask_files):
    mri_path = os.path.join(mri_dir, mri_file)
    mask_path = os.path.join(mask_dir, mask_file)
    
    print(f"Processing {mri_file} with {mask_file}...")
    
    try:
        ## Load MRI image and Mask
        image = sitk.ReadImage(mri_path)
        mask = sitk.ReadImage(mask_path)
        
        ## Check dimensions
        if image.GetDimension() != mask.GetDimension():
            print(f"Adjusting dimensions for {mri_file}...")
            ## If the MRI image is 5D, extract 3D
            if image.GetDimension() == 5:  
                image = sitk.Extract(
                    image,
                    size=[image.GetSize()[0], image.GetSize()[1], image.GetSize()[2]],
                    index=[0, 0, 0]
                )
            else:
                raise ValueError(f"Unexpected image dimension {image.GetDimension()} for {mri_file}")
        
        ## Save the adjusted MRI image
        temp_mri_path = os.path.join(mri_dir, f"temp_{mri_file}")
        sitk.WriteImage(image, temp_mri_path)
        
        ## Extract features
        features = extractor.execute(temp_mri_path, mask_path)
        
        ## Flatten features and convert them to a DataFrame row        
        # Use the filename as the ID
        flat_features = {key: value for key, value in features.items()}
        feature_row = pd.DataFrame([flat_features], index=[mri_file.split('.')[0]])  
        
        ## Add to the main DataFrame
        radiomics_data = pd.concat([radiomics_data, feature_row])
    
    except Exception as e:
        print(f"Error processing {mri_file}: {e}")
        continue  

## Save Radiomics features to CSV
output_path = "radiomics_features.csv"
radiomics_data.to_csv(output_path, index=True)
print(f"Radiomics features saved to {output_path}")
