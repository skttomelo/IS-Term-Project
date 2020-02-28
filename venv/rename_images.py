'''
Required File Structure:

.
├── rename_images.py
└── images
    └── [IMAGE FILES GO HERE]
'''

import os

# Sets user defined prefix on run
prefix = input("Enter Image Prefix: ")
image_number = 0

# Loops though and renames each file in the 'images' folder
for file in os.listdir("images/"):
    # Seperates current file 'name' and 'extension'
    name, extension = os.path.splitext(file)

    # If 'prefix' was previously chosen, skip the file
    if prefix in name:
        image_number += 1
        continue

    new_name = prefix + "_" + str(image_number) + extension
    source = 'images/' + file
    destination = 'images/' + new_name
    os.rename(source, destination)

    image_number += 1
