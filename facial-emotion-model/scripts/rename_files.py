import os

def change_extension(directory):
    # Walk through all files and subdirectories
    for root, _, files in os.walk(directory):
        for filename in files:
            print(filename)
            # Check if the file ends with .JPG
            if filename.endswith(".JPG"):
                # Create the new filename with .jpg extension
                new_filename = filename[:-4] + ".jpg"
                # Get the full paths for the old and new filenames
                old_file = os.path.join(root, filename)
                new_file = os.path.join(root, new_filename)
                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed: {old_file} -> {new_file}")

# Specify the directory to process
directory = "facial-emotion-model/data"

# Call the function to change extensions
change_extension(directory)