import os
from glob import glob

def get_dataset(root_dir):
    """
    Create a list of all BraTS patients and their MRI file paths.
    """

    patient_dirs = sorted(glob(os.path.join(root_dir, "BraTS20_Training_*")))

    dataset = []

    for patient in patient_dirs:

        patient_name = os.path.basename(patient)

        sample = {
            "id": patient_name,
            "t1": os.path.join(patient, patient_name + "_t1.nii"),
            "t2": os.path.join(patient, patient_name + "_t2.nii"),
            "flair": os.path.join(patient, patient_name + "_flair.nii"),
            "t1ce": os.path.join(patient, patient_name + "_t1ce.nii"),
            "seg": os.path.join(patient, patient_name + "_seg.nii")
        }

        # Check if all required files exist
        if all(os.path.exists(path) for path in sample.values() if path != sample["id"]):
            dataset.append(sample)
        else:
            print(f"Skipping {patient_name} - missing files")

    return dataset