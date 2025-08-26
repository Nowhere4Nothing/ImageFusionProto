#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageReslice.h>
#include <vtkImageBlend.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkPointData.h>
#include <vtkDICOMImageReader.h>

namespace py = pybind11;

class VTKSliceExtractor {
public:
    VTKSliceExtractor() {
        reslice = vtkSmartPointer<vtkImageReslice>::New();
        reslice->SetInterpolationModeToLinear();
        reslice->SetAutoCropOutput(1);

        blend = vtkSmartPointer<vtkImageBlend>::New();
        blend->SetOpacity(0, 1.0);
        blend->SetOpacity(1, 0.5);

        transform = vtkSmartPointer<vtkTransform>::New();
        transform->PostMultiply();
    }

    void set_fixed(vtkImageData* img) {
        fixed = img;
        _wire_blend();
        _sync_reslice_output();
    }

    void set_moving(vtkImageData* img) {
        moving = img;
        reslice->SetInputData(moving);
        _apply_transform();
        _wire_blend();
        _sync_reslice_output();
    }

    void set_transform(double tx, double ty, double tz,
                       double rx, double ry, double rz) {
        if (!fixed || !moving) return;
        double center[3];
        fixed->GetCenter(center);

        vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();
        t->PostMultiply();
        t->Translate(-center[0], -center[1], -center[2]);
        t->RotateX(rx);
        t->RotateY(ry);
        t->RotateZ(rz);
        t->Translate(center[0], center[1], center[2]);
        t->Translate(tx, ty, tz);

        transform->DeepCopy(t);
        _apply_transform();
    }

    void set_opacity(double alpha) {
        blend->SetOpacity(1, alpha);
    }

    py::array_t<uint8_t> get_slice(const std::string& orientation, int slice_idx) {
        blend->Update();
        vtkImageData* img = blend->GetOutput();
        int extent[6];
        img->GetExtent(extent);
        int nx = extent[1]-extent[0]+1;
        int ny = extent[3]-extent[2]+1;
        int nz = extent[5]-extent[4]+1;

        uint8_t* out_ptr = new uint8_t[nx*ny]; // max size, will slice later
        int w,h;
        if (orientation == "axial") {
            int z = std::max(0, std::min(slice_idx - extent[4], nz-1));
            for(int j=0;j<ny;j++){
                for(int i=0;i<nx;i++){
                    double* pixel = static_cast<double*>(img->GetScalarPointer(i,j,z));
                    out_ptr[j*nx+i] = static_cast<uint8_t>(*pixel);
                }
            }
            w = nx; h = ny;
        } else if (orientation == "coronal") {
            int y = std::max(0,std::min(slice_idx - extent[2], ny-1));
            for(int k=0;k<nz;k++){
                for(int i=0;i<nx;i++){
                    double* pixel = static_cast<double*>(img->GetScalarPointer(i,y,k));
                    out_ptr[k*nx+i] = static_cast<uint8_t>(*pixel);
                }
            }
            w = nx; h = nz;
        } else if (orientation == "sagittal") {
            int x = std::max(0,std::min(slice_idx - extent[0], nx-1));
            for(int k=0;k<nz;k++){
                for(int j=0;j<ny;j++){
                    double* pixel = static_cast<double*>(img->GetScalarPointer(x,j,k));
                    out_ptr[k*ny+j] = static_cast<uint8_t>(*pixel);
                }
            }
            w = ny; h = nz;
        } else {
            delete[] out_ptr;
            return py::array_t<uint8_t>();
        }

        auto result = py::array_t<uint8_t>({h,w}, out_ptr);
        return result;
    }

private:
    vtkSmartPointer<vtkImageReslice> reslice;
    vtkSmartPointer<vtkImageBlend> blend;
    vtkSmartPointer<vtkTransform> transform;
    vtkImageData* fixed = nullptr;
    vtkImageData* moving = nullptr;

    void _apply_transform() {
        if (!fixed || !moving) return;
        reslice->SetResliceAxes(transform->GetMatrix());
        reslice->Modified();
    }

    void _wire_blend() {
        blend->RemoveAllInputs();
        if (fixed) blend->AddInputData(fixed);
        if (moving) blend->AddInputConnection(reslice->GetOutputPort());
    }

    void _sync_reslice_output() {
        if (!fixed) return;
        double sp[3]; fixed->GetSpacing(sp);
        double org[3]; fixed->GetOrigin(org);
        int ext[6]; fixed->GetExtent(ext);
        reslice->SetOutputSpacing(sp);
        reslice->SetOutputOrigin(org);
        reslice->SetOutputExtent(ext);
        reslice->Modified();
    }
};

PYBIND11_MODULE(vtk_slice_backend, m) {
    py::class_<VTKSliceExtractor>(m, "VTKSliceExtractor")
        .def(py::init<>())
        .def("set_fixed", &VTKSliceExtractor::set_fixed)
        .def("set_moving", &VTKSliceExtractor::set_moving)
        .def("set_transform", &VTKSliceExtractor::set_transform)
        .def("set_opacity", &VTKSliceExtractor::set_opacity)
        .def("get_slice", &VTKSliceExtractor::get_slice);
}
