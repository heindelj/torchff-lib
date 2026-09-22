#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

// Slater-penetrated multipole electrostatics: per-pair energies, the field (dE/dm, the
// coupled-solve matvec) and the first-order VJPs of both. See csrc/slaterelec/slaterelec.cu.
TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("slater_elec_pair_energy(Tensor coords, Tensor pairs, Tensor b, Tensor gate, Tensor m, Tensor n) -> Tensor");
    m.def("slater_elec_pair_grad(Tensor coords, Tensor pairs, Tensor b, Tensor gate, Tensor m, Tensor n, Tensor g) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("slater_elec_field(Tensor coords, Tensor pairs, Tensor b, Tensor gate, Tensor m, Tensor n) -> Tensor");
    m.def("slater_elec_field_vjp(Tensor coords, Tensor pairs, Tensor b, Tensor gate, Tensor m, Tensor n, Tensor lam) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff Slater-penetrated multipole electrostatics";
}
