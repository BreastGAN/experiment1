# Copyright 2018 Lukas Jendele and Ondrej Skopek. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import csv
import os
from dotmap import DotMap


def init(breast_prefix):
    global bcdr_path, inbreast_path
    bcdr_path = os.path.join(breast_prefix, '1_BCDR')
    inbreast_path = os.path.join(breast_prefix, '2_INbreast')


def read_inbreast_csv(fname):
    lines = {}
    with open(fname, 'r') as f:
        next(f)  # skip first line
        for line in csv.reader(f, delimiter=';'):
            line = line[2:]  # skip first two (redacted PatientID, Patient Age)
            laterality, view, date, filename, _, birads, mass, *_ = line  # ignore acr and remainder
            parsed = DotMap()
            parsed.laterality = laterality
            parsed.view = view
            parsed.year = int(date[:4])
            parsed.semester = int(date[4:])
            parsed.fid = int(filename)
            parsed.birads = int(birads[0])
            parsed.cancer = (mass.strip() == 'X')
            if parsed.fid in lines:
                raise ValueError('fid already exists: ' + str(parsed.fid))
            lines[parsed.fid] = parsed
    return lines


def load_inbreast():
    dicom_path = os.path.join(inbreast_path, 'AllDICOMs')
    csv_path = os.path.join(inbreast_path, 'INbreast_2.csv')
    image_metadata = read_inbreast_csv(csv_path)
    patients = {}
    for fname in os.listdir(dicom_path):
        path = os.path.join(dicom_path, fname)
        if not os.path.isfile(path) or not path.endswith('.dcm') or fname.startswith('.'):
            continue

        fid, patient_id, modality, laterality, view, _ = fname[:-4].split('_')
        if view == 'ML':
            view = 'MLO'
        fid = int(fid)

        # Add to images
        assert fid in image_metadata
        cur = image_metadata[fid]
        cur.image_path = path
        assert laterality == cur.laterality
        assert view == cur.view

        # Add to patients
        if patient_id not in patients:
            cur = DotMap()
            cur.patient_id = patient_id
            cur.image_metadata = {}
            patients[patient_id] = cur
        cur = patients[patient_id]
        cur.image_metadata[fid] = image_metadata[fid]

    return image_metadata, patients


def read_bcdr_img_csv(fname):
    lines = {}
    with open(fname, 'r') as f:
        next(f)  # skip first line
        for i, line in enumerate(csv.reader(f, delimiter=',')):
            line = [el.strip() for el in line]
            patient_id, study_id, series, image_filename, image_type_name, image_type_id, age, density = line
            parsed = DotMap()
            parsed.patient_id = int(patient_id)
            parsed.study_id = int(study_id)
            parsed.series = int(series)
            parsed.image_filename = image_filename
            parsed.laterality = image_type_name[0]
            parsed.view = image_type_name[1:]
            parsed.image_type_id = int(image_type_id)
            parsed.age = int(age)
            parsed.density = None if density == 'NaN' else int(density)

            parsed.fid = i
            if parsed.fid in lines:
                raise ValueError('fid already exists: ' + str(parsed.fid))
            lines[parsed.fid] = parsed
    return lines


def read_bcdr_outlines_csv(fname, images):
    for img in images.values():
        img.cancer = False

    with open(fname, 'r') as f:
        next(f)  # skip first line
        for i, line in enumerate(csv.reader(f, delimiter=',')):
            line = [el.strip() for el in line]
            patient_id, study_id, series, lesion_id, segmentation_id, image_view, image_filename, \
                lw_x_points, lw_y_points, mammography_type, mammography_nodule, mammography_calcification, \
                mammography_microcalcification, mammography_axillary_adenopathy, \
                mammography_architectural_distortion, mammography_stroma_distortion, \
                age, density, classification = line

            for key, img in images.items():
                if img.image_filename == image_filename:
                    img.cancer = True

    return images


def load_bcdr(dataset_name):
    dataset_folder = os.path.join(bcdr_path, "{}_dataset".format(dataset_name))
    csv_path = os.path.join(dataset_folder, "{}_img.csv".format(dataset_name.replace(r'-', '_').lower()))
    outline_path = os.path.join(dataset_folder, "{}_outlines.csv".format(dataset_name.replace(r'-', '_').lower()))

    image_metadata = read_bcdr_img_csv(csv_path)
    if os.path.exists(outline_path):
        image_metadata = read_bcdr_outlines_csv(outline_path, image_metadata)

    patients = {}
    for img_meta in image_metadata.values():
        path = os.path.join(dataset_folder, img_meta.image_filename)
        if not os.path.isfile(path) or not path.endswith('.tif') or os.path.basename(path).startswith('.'):
            raise ValueError("File '{}' not valid.".format(path))

        # Add to images
        img_meta.image_path = path

        if img_meta.view == 'O':
            img_meta.view = 'MLO'

        # Add to patients
        patient_id = img_meta.patient_id
        if patient_id not in patients:
            cur = DotMap()
            cur.patient_id = patient_id
            cur.image_metadata = {}
            patients[patient_id] = cur
        cur = patients[patient_id]
        cur.image_metadata[img_meta.fid] = img_meta

    return image_metadata, patients
