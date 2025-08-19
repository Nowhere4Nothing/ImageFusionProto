#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <algorithm>

namespace py = pybind11;

// orientation constants
enum Orientation { AXIAL, CORONAL, SAGITTAL };

// clamp function
inline int clamp(int val, int min_val, int max_val) {
    return std::max(min_val, std::min(max_val, val));
}

py::array_t<uint8_t> get_slice_fast(py::array_t<uint8_t> arr3d, const std::string& orientation_str, int slice_idx) {
    if (arr3d.ndim() != 3) return py::array_t<uint8_t>();

    auto buf = arr3d.unchecked<3>();  // (Z,Y,X)
    int nz = buf.shape(0);
    int ny = buf.shape(1);
    int nx = buf.shape(2);

    Orientation orientation;
    if (orientation_str == "axial") orientation = AXIAL;
    else if (orientation_str == "coronal") orientation = CORONAL;
    else if (orientation_str == "sagittal") orientation = SAGITTAL;
    else return py::array_t<uint8_t>();

    int width = 0, height = 0;
    if (orientation == AXIAL) { width = nx; height = ny; }
    else if (orientation == CORONAL) { width = nx; height = nz; }
    else { width = ny; height = nz; }

    py::array_t<uint8_t> result({height, width});
    auto out = result.mutable_unchecked<2>();

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            uint8_t val = 0;
            switch (orientation) {
                case AXIAL: {
                    int z = clamp(slice_idx,0,nz-1);
                    val = buf(z,i,j);
                    break;
                }
                case CORONAL: {
                    int y = clamp(slice_idx,0,ny-1);
                    val = buf(i,y,j);
                    break;
                }
                case SAGITTAL: {
                    int x = clamp(slice_idx,0,nx-1);
                    val = buf(i,j,x);
                    break;
                }
            }
            // Flip vertically
            out(height-1-i,j) = val;
        }
    }

    return result;
}

PYBIND11_MODULE(vtk_slice_backend, m) {
    m.def("get_slice_fast", &get_slice_fast, "Extract 2D slice from 3D numpy array",
          py::arg("arr3d"), py::arg("orientation_str"), py::arg("slice_idx"));
}
