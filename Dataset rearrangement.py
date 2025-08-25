"""
The purpose of this script is to rearrange the dataset
into a more convenient way to read.
"""

# import the package
from pathlib import Path
import shutil

Ori_dataset = Path('Dataset') # The existed folder
Tar_dataset = Path('Dataset-processed') # The folder contains rearranged data
stages = ['Proof_platelet_I', 'Proof_platelet_II', 'Proof_platelet_III', 'Proof_platelet_IV']
cleaned_stages = ['StageI', 'StageII', 'StageIII', 'StageIV']

# create new subfolder in target folder
for stage in cleaned_stages:
    (Tar_dataset / stage).mkdir(parents = True, exist_ok = True) # the parent dictionary will be established if needed

for ori_stage, new_stage in zip(stages, cleaned_stages):
    raw_material_path = Ori_dataset / ori_stage / 'raw_micrographs' # the .mrc file
    mask_path = Ori_dataset / ori_stage / 'masks' # the .svg file
    output_path = Tar_dataset / new_stage

    # sort the .mrc files
    mrc_files = sorted(raw_material_path.glob('*.mrc'))

    # for each mrc file
    for idx, file in enumerate(mrc_files, start = 1):
        new_name = f"{new_stage}-{idx: 03d}" # the first file in stageI is: stageI-001.mrc
        related_svg = mask_path / file.stem / "000001.svg" # find the corresponding .svg file of the mrc file

        if not related_svg.exists(): # If that .svg file doesn't exist
            print(f"The mask {related_svg} for the original mrc file {file} does not exist.")
            continue

        # copy the mrc and svg to the new folder
        shutil.copy2(file, output_path / f"{new_name}.mrc")
        shutil.copy2(related_svg, output_path / f"{new_name}.svg")

print("The data rearrangement work is completed.")