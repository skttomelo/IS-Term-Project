import os

# Can be referenced to list all subdirectories of a given directory
def get_subdirectories(directory):
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]

# If number of subdirectories equals zero...
if len(get_subdirectories(os.getcwd())) == 0:
    print("\nNo valid directories exist. Create a directory and run again.")
    quit()
else:
    # Get user defined prefix in lowercase text
    prefix = input("\nEnter image prefix: ").lower()

    # Format and print subdirectories
    print("\nDirectories: ")
    for subdirectory in get_subdirectories(os.getcwd()):
        print("   " + subdirectory + "/")

# Creates a list of available directories in lowercase text
directory_list = [x for x in input("\nEnter one or more of the above directories: ").lower().split()]
print("")

# Loops though and renames each file in each chosen directory
for selected_directory in directory_list:

    # Resets image counter for each directory
    image_number = 0

    try:
        for file in os.listdir(selected_directory):
            # Seperates current file name and extension
            name, extension = os.path.splitext(file)

            # If the same prefix was previously chosen, skip the file
            if prefix in name:
                image_number += 1
                continue

            # Formats file name and extension before renaming
            new_name = prefix + "_" + str(image_number) + extension
            source = selected_directory + file
            destination = selected_directory + new_name
            os.rename(source, destination)

            image_number += 1

    # Prints directory if misspelled or missing
    except:
        print("Error: " + selected_directory + " does not exist.")

    # Outputs number of files renamed in chosen directories
    print("   " + selected_directory + "   \t" + str(image_number) + " files renamed.")
