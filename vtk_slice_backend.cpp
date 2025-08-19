#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vtkSmartPointer.h>
#include <vtkDICOMImageReader.h>
#include <vtkImageReslice.h>
#include <vtkImageBlend.h>
#include <vtkTransform.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkMatrix4x4.h>
#include <vtkImageCast.h>
#include <vtkImageShiftScale.h>
#include <string>
#include <algorithm>

namespace py = pybind11;

class VTKEngine {
public:
    VTKEngine() {
        transform->PostMultiply();
        reslice->SetInterpolationModeToLinear();
        reslice->SetBackgroundLevel(0.0);
        blend->SetOpacity(0, 1.0);
        blend->SetOpacity(1, 0.5);
    }

    bool load_fixed(const std::string &dir) {
        auto reader_ = vtkSmartPointer<vtkDICOMImageReader>::New();
        reader_->SetDirectoryName(dir.c_str());
        reader_->Update();
        fixed_reader = reader_;
        wire_blend();
        sync_reslice();
        return true;
    }

    bool load_moving(const std::string &dir) {
        auto reader_ = vtkSmartPointer<vtkDICOMImageReader>::New();
        reader_->SetDirectoryName(dir.c_str());
        reader_->Update();
        moving_reader = reader_;
        reslice->SetInputConnection(reader_->GetOutputPort());
        apply_transform();
        wire_blend();
        sync_reslice();
        return true;
    }

    void set_translation(double tx_, double ty_, double tz_) {
        tx = tx_; ty = ty_; tz = tz_;
        apply_transform();
    }

    void set_rotation(double rx_, double ry_, double rz_) {
        rx = rx_; ry = ry_; rz = rz_;
        apply_transform();
    }

    void set_opacity(double alpha) {
        blend->SetOpacity(1, std::clamp(alpha,0.0,1.0));
        blend->Modified();
    }

    void reset_transform() {
        tx = ty = tz = rx = ry = rz = 0.0;
        transform->Identity();
        apply_transform();
    }

    py::array_t<uint8_t> get_slice(const std::string &orientation_str, int slice_idx) {
        if (!fixed_reader) return py::array_t<uint8_t>();

        blend->Modified();
        blend->Update();

        vtkImageData *img = blend->GetOutput();
        int ext[6]; img->GetExtent(ext);
        int nx = ext[1]-ext[0]+1;
        int ny = ext[3]-ext[2]+1;
        int nz = ext[5]-ext[4]+1;

        auto scalars = img->GetPointData()->GetScalars();
        if (!scalars) return py::array_t<uint8_t>();

        py::ssize_t width=0, height=0;
        std::vector<uint8_t> buffer;

        if (orientation_str=="axial") {
            width = nx; height = ny;
            buffer.resize(width*height);
            int z = std::clamp(slice_idx - ext[4], 0, nz-1);
            for(int i=0;i<height;i++)
                for(int j=0;j<width;j++)
                    buffer[(height-1-i)*width + j] = scalars->GetComponent(z*ny*nx + i*nx + j,0);
        }
        else if (orientation_str=="coronal") {
            width = nx; height = nz;
            buffer.resize(width*height);
            int y = std::clamp(slice_idx - ext[2], 0, ny-1);
            for(int i=0;i<height;i++)
                for(int j=0;j<width;j++)
                    buffer[(height-1-i)*width + j] = scalars->GetComponent(i*ny*nx + y*nx + j,0);
        }
        else if (orientation_str=="sagittal") {
            width = ny; height = nz;
            buffer.resize(width*height);
            int x = std::clamp(slice_idx - ext[0], 0, nx-1);
            for(int i=0;i<height;i++)
                for(int j=0;j<width;j++)
                    buffer[(height-1-i)*width + (width-1-j)] = scalars->GetComponent(i*ny*nx + j*nx + x,0);
        }
        else return py::array_t<uint8_t>();

        return py::array_t<uint8_t>({height,width}, buffer.data());
    }

private:
    vtkSmartPointer<vtkDICOMImageReader> fixed_reader=nullptr;
    vtkSmartPointer<vtkDICOMImageReader> moving_reader=nullptr;
    vtkSmartPointer<vtkImageReslice> reslice=vtkSmartPointer<vtkImageReslice>::New();
    vtkSmartPointer<vtkImageBlend> blend=vtkSmartPointer<vtkImageBlend>::New();
    vtkSmartPointer<vtkTransform> transform=vtkSmartPointer<vtkTransform>::New();

    double tx=0, ty=0, tz=0;
    double rx=0, ry=0, rz=0;

    void apply_transform() {
        if (!fixed_reader || !moving_reader) return;

        double center[3];
        fixed_reader->GetOutput()->GetCenter(center);

        vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();
        t->PostMultiply();
        t->Translate(-center[0], -center[1], -center[2]);
        t->RotateX(rx); t->RotateY(ry); t->RotateZ(rz);
        t->Translate(center[0]+tx, center[1]+ty, center[2]+tz);

        transform->DeepCopy(t);
        reslice->SetResliceAxes(transform->GetMatrix());
        reslice->Modified();
    }

    void wire_blend() {
        blend->RemoveAllInputs();
        if(fixed_reader) blend->AddInputConnection(fixed_reader->GetOutputPort());
        if(moving_reader) blend->AddInputConnection(reslice->GetOutputPort());
        blend->Modified();
    }

    void sync_reslice() {
        if(!fixed_reader) return;
        vtkImageData* img = fixed_reader->GetOutput();
        reslice->SetOutputSpacing(img->GetSpacing());
        reslice->SetOutputOrigin(img->GetOrigin());
        reslice->SetOutputExtent(img->GetExtent());
        reslice->Modified();
    }
};

PYBIND11_MODULE(vtk_engine, m) {
    py::class_<VTKEngine>(m, "VTKEngine")
        .def(py::init<>())
        .def("load_fixed",&VTKEngine::load_fixed)
        .def("load_moving",&VTKEngine::load_moving)
        .def("set_translation",&VTKEngine::set_translation)
        .def("set_rotation",&VTKEngine::set_rotation)
        .def("set_opacity",&VTKEngine::set_opacity)
        .def("reset_transform",&VTKEngine::reset_transform)
        .def("get_slice",&VTKEngine::get_slice);
}
