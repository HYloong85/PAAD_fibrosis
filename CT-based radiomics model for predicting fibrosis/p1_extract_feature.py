from __future__ import print_function
import six
import os  # For file system operations
import numpy as np
import radiomics
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor  # Main feature extraction module
import argparse  # Currently unused but kept for potential CLI expansion


def extract_radiomics_features(imagePath, maskPath):
    """
    Extract radiomic features from medical images using PyRadiomics

    Args:
        imagePath (str): Path to medical image (e.g., CT, MRI)
        maskPath (str): Path to segmentation mask

    Returns:
        tuple: (feature_values, feature_names)
    """
    # Validate inputs
    if not (os.path.exists(imagePath) and os.path.exists(maskPath)):
        raise FileNotFoundError('Image or mask file not found')

    # Configure extraction parameters
    settings = {
        'binWidth': 25,  # Histogram bin width
        'sigma': [2, 3, 5],  # Sigma values for LoG filter
        'Interpolator': sitk.sitkBSpline,  # Resampling interpolator
        'resampledPixelSpacing': [1, 1, 1],  # Isotropic resampling spacing
        'voxelArrayShift': 1000,  # Intensity shift parameter
        'normalize': True,  # Enable intensity normalization
        'normalizeScale': 100  # Normalization scaling factor
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    print(f'Extraction parameters:\n\t{extractor.settings}')

    # Enable specific image pre-processing filters
    extractor.enableImageTypeByName('LoG')  # Laplacian of Gaussian
    extractor.enableImageTypeByName('Wavelet')  # Wavelet transform

    # Selective feature class configuration
    extractor.enableFeaturesByName(
        firstorder=[
            'Energy', 'TotalEnergy', 'Entropy', 'Minimum',
            '10Percentile', '90Percentile', 'Maximum', 'Mean',
            'Median', 'InterquartileRange', 'Range',
            'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation',
            'RootMeanSquared', 'StandardDeviation', 'Skewness',
            'Kurtosis', 'Variance', 'Uniformity'
        ],
        shape=[
            'VoxelVolume', 'MeshVolume', 'SurfaceArea',
            'SurfaceVolumeRatio', 'Compactness1', 'Compactness2',
            'Sphericity', 'SphericalDisproportion', 'Maximum3DDiameter',
            'Maximum2DDiameterSlice', 'Maximum2DDiameterColumn',
            'Maximum2DDiameterRow', 'MajorAxisLength',
            'MinorAxisLength', 'LeastAxisLength', 'Elongation',
            'Flatness'
        ]
    )

    print(f'Enabled filters:\n\t{extractor.enabledImagetypes}')

    # Execute feature extraction
    result = extractor.execute(imagePath, maskPath, label=1)

    # Process results
    feature_values = []
    feature_names = []

    # Skip first 37 generic features (per PyRadiomics output structure)
    SKIP_INITIAL = 37

    for i, (key, value) in enumerate(six.iteritems(result)):
        if i >= SKIP_INITIAL:
            feature_names.append(key)
            try:
                feature_values.append(float(value))  # Ensure numeric conversion
            except (TypeError, ValueError):
                feature_values.append(np.nan)  # Handle invalid values
                print(f'WARNING: Non-convertible feature value for {key}: {value}')

    return np.array(feature_values), np.array(feature_names)


# Main processing routine
if __name__ == "__main__":
    # Configuration
    IMAGE_DIR = './images'  # Directory containing input images
    MASK_DIR = './mask'  # Directory containing segmentation masks
    OUTPUT_FILE = './Radiomics-features_1145.xlsx'  # Output Excel path

    # Initialize storage
    all_features = []
    patient_ids = []

    # Process each patient
    for image_file in os.listdir(IMAGE_DIR):
        patient_id = os.path.splitext(image_file)[0]
        print(f'Processing: {patient_id}')

        # Find matching mask file (adjust pattern as needed)
        mask_match = next((m for m in os.listdir(MASK_DIR) if m.startswith(patient_id.split('_')[0])), None)

        if not mask_match:
            print(f'WARNING: No mask found for {patient_id}')
            continue

        image_path = os.path.join(IMAGE_DIR, image_file)
        mask_path = os.path.join(MASK_DIR, mask_match)

        print(f'\tImage: {image_path}')
        print(f'\tMask: {mask_path}')

        try:
            # Extract features
            features, feature_names = extract_radiomics_features(image_path, mask_path)

            # Store results
            all_features.append(features)
            patient_ids.append(patient_id)

            # Print progress
            print(f'\tExtracted {len(features)} features | Current total: {len(all_features)} patients')

        except Exception as e:
            print(f'ERROR processing {patient_id}: {str(e)}')

    # Create DataFrame
    feature_df = pd.DataFrame(all_features, index=patient_ids, columns=feature_names)

    # Save results
    print(f'Saving {len(patient_ids)} records to {OUTPUT_FILE}')
    feature_df.to_excel(OUTPUT_FILE)
    print('Processing complete!')