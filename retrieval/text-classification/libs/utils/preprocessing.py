import SimpleITK as sitk
import argparse
import os
import glob
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re

FIXED_DIRECTION = [1, 0, 0, 0, 1, 0, 0, 0]


def get_patients(path):
    assert path is not None, "Missing the input path!"
    assert path.is_dir(), 'Invalid input path!'

    regex = re.compile(r'.*volume-(?P<id>[0-9]+).*')

    def get_sort_key(x):
        _id = regex.match(str(x)).group('id')
        return int(_id)

    patients = map(str, path.glob('*volume-*'))
    patients = sorted(patients, key=get_sort_key)
    return patients


def get_nii_image(img_path):
    image = sitk.ReadImage(img_path)

    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkInt16)
    image = castFilter.Execute(image)

    return image


def read_information(filename):
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.ReadImageInformation()
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        print("({0}) = = \"{1}\"".format(k, v))


def set_series_tag_values(nii_image, name):
    # direction = nii_image.GetDirection()
    direction = FIXED_DIRECTION
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    series_tag_values = [("0008|0031", modification_time),  # Series Time
                         ("0008|0021", modification_date),  # Series Date
                         ("0008|0008", "ORIGINAL\\PRIMARY"),  # Image Type
                         ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date + \
                          ".1"+modification_time),  # Series Instance UID
                         ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                                           direction[1], direction[4], direction[7])))),
                         ("0008|103e", name)]  # Series Description
    return series_tag_values


def writeSlices(img_serie, series_tag_values, i, final_folder):
    image_slice = img_serie[:, :, i]

    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(
        tag_value[0], tag_value[1]), series_tag_values))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime(
        "%Y%m%d"))  # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime(
        "%H%M%S"))  # Instance Creation Time

    # Setting the type to CT preserves the slice location.
    # set the type to CT so the thickness is carried over
    image_slice.SetMetaData("0008|0060", "CT")

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData("0020|0032", '\\'.join(map(
        str, img_serie.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
    image_slice.SetMetaData("0020,0013", str(i))  # Instance Number

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(os.path.join(final_folder / f'{i:03d}.dcm'))
    writer.Execute(image_slice)


def convert(_root, _output, _input, _type):
    print(
        f'Working on ({_root}/{_input}, {_type}), output to {_root}/{_output}')
    root = Path(_root)
    patients = get_patients(root / _input)
    output_path = root / _output / _input
    output_path.mkdir(parents=True, exist_ok=True)

    for patient in tqdm(patients):
        patient_id = int(patient.split('-')[-1].split('.')[0])
        patient_folder = output_path / f'LiTS{patient_id:03d}' / _type
        patient_folder.mkdir(parents=True, exist_ok=True)

        img_path = patient if _type == 'PATIENT' \
            else patient.replace("volume", "segmentation")
        nii_image = get_nii_image(img_path)
        # read_information(img_path)

        series_tag_values = set_series_tag_values(
            nii_image, f'{_type}_{patient_id}')

        list(map(lambda k: writeSlices(img_serie=nii_image, series_tag_values=series_tag_values,
                                       i=k, final_folder=patient_folder), range(nii_image.GetDepth())))


def test():
    path = '../Data/LiTS/LiTS_train/segmentation-38.nii'
    img = sitk.ReadImage(path)
    if img.GetPixelID() > 4:
        print("ahihi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--output')
    parser.add_argument('--input', default=None)
    parser.add_argument('--type', default=None)
    args = parser.parse_args()

    if args.input is None:
        for _input in ['train', 'val']:
            if args.type is None:
                for _type in ['PATIENT', 'LABEL']:
                    convert(args.root, args.output, _input, _type)
            else:
                convert(args.root, args.output, _input, args.type)
    else:
        if args.type is None:
            for _type in ['PATIENT', 'LABEL']:
                convert(args.root, args.output, args.input, _type)
        else:
            convert(args.root, args.output, args.input, args.type)
