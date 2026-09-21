#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

// Per-element (per pair / bond / angle) force-field terms. Output is unreduced; autograd,
// including double backward, is assembled in Python (torchff/ffterms.py) from the
// _energy / _grad / _hvp triples registered here. See csrc/ffterms/ffterms.cu.
TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("tt_dispersion_pair_energy(Tensor coords, Tensor pairs, Tensor c6, Tensor b) -> Tensor");
    m.def("tt_dispersion_pair_grad(Tensor coords, Tensor pairs, Tensor c6, Tensor b, Tensor g) -> (Tensor, Tensor, Tensor)");
    m.def("tt_dispersion_pair_hvp(Tensor coords, Tensor pairs, Tensor c6, Tensor b, Tensor g, Tensor v_coords, Tensor v_c6, Tensor v_b) -> (Tensor, Tensor, Tensor, Tensor)");

    m.def("morse_bond_energy(Tensor coords, Tensor bonds, Tensor r_eq, Tensor d, Tensor k) -> Tensor");
    m.def("morse_bond_grad(Tensor coords, Tensor bonds, Tensor r_eq, Tensor d, Tensor k, Tensor g) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("morse_bond_hvp(Tensor coords, Tensor bonds, Tensor r_eq, Tensor d, Tensor k, Tensor g, Tensor v_coords, Tensor v_req, Tensor v_d, Tensor v_k) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    m.def("cosine_angle_energy(Tensor coords, Tensor angles, Tensor cos_eq, Tensor k) -> Tensor");
    m.def("cosine_angle_grad(Tensor coords, Tensor angles, Tensor cos_eq, Tensor k, Tensor g) -> (Tensor, Tensor, Tensor)");
    m.def("cosine_angle_hvp(Tensor coords, Tensor angles, Tensor cos_eq, Tensor k, Tensor g, Tensor v_coords, Tensor v_cos, Tensor v_k) -> (Tensor, Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff per-element force-field terms (double-backward capable)";
}
