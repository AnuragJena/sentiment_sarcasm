#Giving Permission to notebook to access Drive
from google.colab import drive
import os

# Function to mount Google Drive and handle errors
def mount_drive():
    try:
        print("Mounting Google Drive...")
        drive.mount('/content/drive')  # Attempt to mount Google Drive

        # Check if the mount was successful by verifying the existence of a file in the drive
        drive_check_path = '/content/drive/My Drive/Colab Files/Output/'
        if not os.path.exists(drive_check_path):
            raise FileNotFoundError(f"Google Drive is not mounted properly. Please check for the drive path {drive_check_path} and try again.")

        print("Google Drive mounted successfully!")
    except Exception as e:
        print(f"Error while mounting Google Drive: {e}")

# Call the function to mount Google Drive
mount_drive()