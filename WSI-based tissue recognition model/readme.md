#### m1_extract_patch.m

We use imagescope x64 to mark different organizational regions and generate XML annotations. Use this script to extract various types of labeled tissue image blocks.

#### m2_train_classifier.m

This script is used to train the tissue recognition model

#### m3_deploy_classifier.m

This script performs automated tissue recognition on whole-slide images (or specified ROIs), outputting area proportion values for each tissue class.

#### **Survival Analysis**.R

This script is used for survival analysis of high and low stroma proportion groups.

**output_images**   saves the WSIs segmentation map visualization results of the tissue recognition model.

**output_masks** and **output_rgbout**   saves the WSI classification result matrix and RGB weighting matrix of the organization identification model

**Subroutines** contains subroutines needed to run scripts