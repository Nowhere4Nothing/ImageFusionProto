import os
import re
import copy
import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, MULTIPLE

import numpy as np
import pydicom
import pydicom.uid
import SimpleITK as sitk


class DICOMROITransferApp:
    """
    GUI tool that appends transformed ROIs from a source RTSTRUCT into an
    existing target RTSTRUCT file (in-place).
    """
    def __init__(self, root):
        self.root = root
        self.root.title("DICOM ROI Append Tool (append into existing RTSTRUCT)")

        # filepaths / datasets
        self.source_folder = None
        self.target_folder = None
        self.source_rtss_path = None
        self.target_rtss_path = None
        self.source_rtss = None
        self.target_rtss = None
        self.transform_path = None
        self.transform = None

        # target image slice info
        self.target_sop_uids = []       # list of SOPInstanceUIDs
        self.target_sop_uids_class = [] # corresponding SOPClassUIDs
        self.target_z_positions = []    # corresponding z positions

        # UI
        tk.Button(root, text="Load Source ImageSet Folder", command=self.load_source_folder).pack(pady=3)
        tk.Button(root, text="Load Target ImageSet Folder", command=self.load_target_folder).pack(pady=3)
        tk.Button(root, text="Load Source RTSTRUCT (with ROIs to copy)", command=self.load_source_rtss).pack(pady=3)
        tk.Button(root, text="Load Target RTSTRUCT (to append into)", command=self.load_target_rtss).pack(pady=3)
        tk.Button(root, text="Load Transform (.txt)", command=self.load_transform).pack(pady=3)

        tk.Label(root, text="Select ROIs from source to append:").pack(anchor="w", padx=6)
        self.roi_listbox = Listbox(root, selectmode=MULTIPLE, width=60, height=12)
        self.roi_listbox.pack(padx=6, pady=6)

        tk.Button(root, text="Start Transfer (append into selected target RTSTRUCT)", command=self.start_transfer).pack(pady=8)

    # ---------------- file loaders ----------------
    def load_source_folder(self):
        folder = filedialog.askdirectory(title="Select Source ImageSet Folder")
        if not folder:
            return
        self.source_folder = folder
        messagebox.showinfo("Loaded", f"Source image folder set: {folder}")

    def load_target_folder(self):
        folder = filedialog.askdirectory(title="Select Target ImageSet Folder")
        if not folder:
            return
        self.target_folder = folder
        self._build_target_slice_list()
        messagebox.showinfo("Loaded", f"Target image folder set: {folder}\nFound {len(self.target_sop_uids)} image slices.")

    def load_source_rtss(self):
        path = filedialog.askopenfilename(title="Select Source RTSTRUCT file", filetypes=[("DICOM", "*.dcm"), ("All", "*.*")])
        if not path:
            return
        ds = pydicom.dcmread(path)
        # basic validation
        if not hasattr(ds, "StructureSetROISequence") or not hasattr(ds, "ROIContourSequence"):
            messagebox.showerror("Error", "Selected file doesn't look like a valid RTSTRUCT.")
            return
        self.source_rtss_path = path
        self.source_rtss = ds
        self._populate_source_rois()
        messagebox.showinfo("Loaded", f"Source RTSTRUCT loaded: {path}")

    def load_target_rtss(self):
        path = filedialog.askopenfilename(title="Select Target RTSTRUCT file to append into", filetypes=[("DICOM", "*.dcm"), ("All", "*.*")])
        if not path:
            return
        ds = pydicom.dcmread(path)
        if not hasattr(ds, "StructureSetROISequence") or not hasattr(ds, "ROIContourSequence"):
            messagebox.showerror("Error", "Selected file doesn't look like a valid RTSTRUCT.")
            return
        self.target_rtss_path = path
        self.target_rtss = ds
        # build target slice list if not already
        if self.target_folder:
            self._build_target_slice_list()
        messagebox.showinfo("Loaded", f"Target RTSTRUCT loaded (will be appended in-place): {path}")

    def load_transform(self):
        path = filedialog.askopenfilename(title="Select transform.txt (4x4)", filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not path:
            return
        try:
            self.transform = self._read_transform_text_to_sitk(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read transform file:\n{e}")
            return
        self.transform_path = path
        # print short summary
        mat = np.array(self.transform.GetMatrix()).reshape(3, 3)
        tr = np.array(self.transform.GetTranslation())
        print("Loaded transform:")
        print(mat)
        print("translation:", tr)
        messagebox.showinfo("Loaded", f"Transform loaded: {path}")

    # ---------------- helpers ----------------
    def _populate_source_rois(self):
        self.roi_listbox.delete(0, tk.END)
        names = []
        for roi in self.source_rtss.StructureSetROISequence:
            name = str(roi.ROIName)
            names.append(name)
            self.roi_listbox.insert(tk.END, name)
        print(f"Source RTSTRUCT ROIs: {names}")

    def _build_target_slice_list(self):
        """Read target folder and build lists of SOPInstanceUID and slice z-positions (skip non-image files)."""
        if not self.target_folder:
            return
        files = sorted([f for f in os.listdir(self.target_folder) if f.lower().endswith(".dcm")])
        uids = []
        uids_class = []
        zs = []
        for f in files:
            p = os.path.join(self.target_folder, f)
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
                # require ImagePositionPatient and SOPInstanceUID
                if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "SOPInstanceUID"):
                    uids.append(ds.SOPInstanceUID)
                    uids_class.append(getattr(ds, "SOPClassUID", ""))
                    zs.append(float(ds.ImagePositionPatient[2]))
                else:
                    # skip non-image file
                    continue
            except Exception as e:
                print(f"Skipping file {f}: {e}")
        if not uids:
            messagebox.showwarning("Warning", "No valid image slices found in target folder; contour ref mapping will be limited.")
        self.target_sop_uids = uids
        self.target_sop_uids_class = uids_class
        self.target_z_positions = np.array(zs)

    def _read_transform_text_to_sitk(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln for ln in (l.strip() for l in f.readlines()) if ln]
        # sanitize unicode minus / zero-width etc.
        cleaned = []
        for ln in lines:
            ln2 = ln.replace('−', '-')
            ln2 = re.sub(r'[^\d\.\-\seE]', ' ', ln2)
            cleaned.append(" ".join(ln2.split()))
        matrix = np.array([[float(x) for x in ln.split()] for ln in cleaned])
        if matrix.shape != (4, 4):
            raise ValueError("Transform file must be a 4x4 matrix.")
        t = sitk.AffineTransform(3)
        t.SetMatrix(matrix[:3, :3].flatten())
        t.SetTranslation(matrix[:3, 3])
        return t.GetInverse()

    # ---------------- main operation ----------------
    def start_transfer(self):
        # validate prerequisites
        if self.source_rtss is None:
            messagebox.showerror("Error", "Load the source RTSTRUCT first (with the ROIs to copy).")
            return
        if self.target_rtss is None:
            messagebox.showerror("Error", "Load the target RTSTRUCT to append into.")
            return
        if self.transform is None:
            messagebox.showerror("Error", "Load the 4x4 transform file first.")
            return

        sel = self.roi_listbox.curselection()
        if not sel:
            messagebox.showerror("Error", "Select one or more ROIs to transfer from the source RTSTRUCT.")
            return

        selected_names = [self.roi_listbox.get(i) for i in sel]
        print("Selected ROIs to append:", selected_names)

        # Ensure sequences exist in target; if missing create empty python lists
        if not hasattr(self.target_rtss, "StructureSetROISequence"):
            self.target_rtss.StructureSetROISequence = []
        if not hasattr(self.target_rtss, "ROIContourSequence"):
            self.target_rtss.ROIContourSequence = []
        if not hasattr(self.target_rtss, "RTROIObservationsSequence") and hasattr(self.source_rtss, "RTROIObservationsSequence"):
            # create empty and we'll append mapped observations if source has them
            self.target_rtss.RTROIObservationsSequence = []

        # compute next ROI number to use
        existing_nums = [int(getattr(r, "ROINumber", 0)) for r in self.target_rtss.StructureSetROISequence]
        next_roi_number = (max(existing_nums) + 1) if existing_nums else 1

        # existing ROI names set
        existing_names = set(str(r.ROIName) for r in self.target_rtss.StructureSetROISequence)

        # mapping from source roi number -> source sequences for quick lookup
        src_roinum_by_name = {}
        for item in self.source_rtss.StructureSetROISequence:
            src_roinum_by_name[str(item.ROIName)] = int(item.ROINumber)

        # optional mapping of RTROIObservations in source by ROINumber
        src_obs_by_roinum = {}
        if hasattr(self.source_rtss, "RTROIObservationsSequence"):
            for obs in self.source_rtss.RTROIObservationsSequence:
                if hasattr(obs, "ReferencedROINumber"):
                    src_obs_by_roinum[int(obs.ReferencedROINumber)] = obs

        # iterate selected ROI names
        for name in selected_names:
            if name not in src_roinum_by_name:
                print(f"Source ROI '{name}' not found (skipping).")
                continue
            src_num = src_roinum_by_name[name]

            # find ROIContourSequence item in source
            src_roi_contour_item = None
            for rc in self.source_rtss.ROIContourSequence:
                if int(rc.ReferencedROINumber) == src_num:
                    src_roi_contour_item = rc
                    break
            if src_roi_contour_item is None:
                print(f"No contour entry for ROI '{name}' (skipping).")
                continue

            # ensure unique name in target
            new_name = name
            count = 1
            while new_name in existing_names:
                new_name = f"{name}_{count}"
                count += 1
            existing_names.add(new_name)

            # create and append StructureSetROISequence entry (copy from source but update fields)
            # copy source ROI definition if available
            src_ssroi = None
            for s in self.source_rtss.StructureSetROISequence:
                if int(s.ROINumber) == src_num:
                    src_ssroi = s
                    break

            if src_ssroi is not None:
                new_ssroi = copy.deepcopy(src_ssroi)
            else:
                # minimal entry
                new_ssroi = pydicom.dataset.Dataset()
                new_ssroi.ROINumber = next_roi_number

            new_ssroi.ROINumber = next_roi_number
            new_ssroi.ROIName = new_name
            # keep other source metadata if present; don't change StudyInstanceUID etc.
            self.target_rtss.StructureSetROISequence.append(new_ssroi)

            # create RTROIObservationsSequence entry if source has one, otherwise try minimal
            if hasattr(self.target_rtss, "RTROIObservationsSequence") is False:
                self.target_rtss.RTROIObservationsSequence = []

            if src_num in src_obs_by_roinum:
                src_obs = src_obs_by_roinum[src_num]
                new_obs = copy.deepcopy(src_obs)
                # update referenced ROI number to new number
                if hasattr(new_obs, "ReferencedROINumber"):
                    new_obs.ReferencedROINumber = next_roi_number
                # update label if present
                if hasattr(new_obs, "ROIObservationLabel"):
                    new_obs.ROIObservationLabel = new_name
                self.target_rtss.RTROIObservationsSequence.append(new_obs)
            else:
                # create minimal observation dataset
                new_obs = pydicom.dataset.Dataset()
                new_obs.ReferencedROINumber = next_roi_number
                new_obs.ROIObservationLabel = new_name
                self.target_rtss.RTROIObservationsSequence.append(new_obs)

            # create ROIContourSequence entry (deepcopy then update)
            new_roi_contour = copy.deepcopy(src_roi_contour_item)
            new_roi_contour.ReferencedROINumber = next_roi_number

            # transform each contour in new_roi_contour.ContourSequence
            for contour_item in new_roi_contour.ContourSequence:
                if not hasattr(contour_item, "ContourData"):
                    continue
                pts = np.array(contour_item.ContourData, dtype=float).reshape(-1, 3)
                # apply transform (SimpleITK)
                transformed = [self.transform.TransformPoint(tuple(p)) for p in pts]
                contour_item.ContourData = [float(x) for p in transformed for x in p]
                contour_item.NumberOfContourPoints = len(transformed)

                # assign ContourImageSequence referencing closest target slice (if available)
                if self.target_sop_uids and len(self.target_z_positions) == len(self.target_sop_uids):
                    z_mean = np.mean([p[2] for p in transformed])
                    idx = int(np.abs(self.target_z_positions - z_mean).argmin())
                    sop_uid = self.target_sop_uids[idx]
                    sop_class = self.target_sop_uids_class[idx] if idx < len(self.target_sop_uids_class) else None
                    cis_item = pydicom.dataset.Dataset()
                    if sop_class:
                        cis_item.ReferencedSOPClassUID = sop_class
                    cis_item.ReferencedSOPInstanceUID = sop_uid
                    contour_item.ContourImageSequence = [cis_item]
                else:
                    # leave ContourImageSequence unchanged if we can't map
                    pass

            # append new_roi_contour into target RTSTRUCT
            if not hasattr(self.target_rtss, "ROIContourSequence"):
                self.target_rtss.ROIContourSequence = []
            self.target_rtss.ROIContourSequence.append(new_roi_contour)

            print(f"Appended ROI '{new_name}' as ROINumber {next_roi_number} into target RTSTRUCT.")
            next_roi_number += 1

        # At this point, self.target_rtss object in memory has been modified (appended). We will save it back
        # to the same file path selected by the user (overwrite with appended content).
        try:
            # save to same file (overwrite)
            self.target_rtss.save_as(self.target_rtss_path)
            messagebox.showinfo("Success", f"Appended ROIs saved into target RTSTRUCT:\n{self.target_rtss_path}")
            print("Saved appended RTSTRUCT in-place:", self.target_rtss_path)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save updated RTSTRUCT:\n{e}")
            print("Failed to save updated RTSTRUCT:", e)


if __name__ == "__main__":
    root = tk.Tk()
    app = DICOMROITransferApp(root)
    root.mainloop()
