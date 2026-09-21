// Host build of csrc/ffterms/terms.cuh: evaluates each term on Dual<double> and prints
// energy, gradient and Hessian-vector product for fixed inputs. tests/test_ffterms.py
// compiles this with g++ (no CUDA needed) and compares against torch double backward,
// so the device math is checked on any machine.
#define __device__
#define __forceinline__ inline
#include <cmath>
#include <cstdio>
#include "common/dual.cuh"
#include "ffterms/terms.cuh"
using D = Dual<double>;

static void dump(const char* tag, D e, D* g, int n) {
    printf("%s e %.17e %.17e\n", tag, e.v, e.d);
    for (int k = 0; k < n; ++k) printf("%s g%d %.17e %.17e\n", tag, k, g[k].v, g[k].d);
}

int main() {
    {   // Tang-Toennies, direct branch (u ~ 4.4) and series branch (u ~ 1.3)
        double x[5] = {1.3, -0.7, 2.1, 30.0, 1.7}, v[5] = {0.3, 0.2, -0.5, 0.7, -0.4};
        for (int trial = 0; trial < 2; ++trial) {
            if (trial == 1) x[4] = 0.5;
            D e, g[5];
            tt_dispersion_term<D, double>(D(x[0], v[0]), D(x[1], v[1]), D(x[2], v[2]), D(x[3], v[3]), D(x[4], v[4]), e, g);
            dump(trial == 0 ? "TT0" : "TT1", e, g, 5);
        }
    }
    {   double y[6] = {0.9, -0.4, 1.2, 1.81, 0.2, 0.54}, w[6] = {0.1, 0.3, -0.2, 0.05, 0.02, 0.03};
        D e, g[6];
        morse_term<D, double>(D(y[0], w[0]), D(y[1], w[1]), D(y[2], w[2]), D(y[3], w[3]), D(y[4], w[4]), D(y[5], w[5]), e, g);
        dump("MO", e, g, 6);
    }
    {   double y[8] = {1.0, 0.2, -0.3, -0.4, 1.1, 0.5, -0.25, 0.17}, w[8] = {0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.04, 0.02};
        D e, g[8];
        cosine_angle_term<D, double>(D(y[0], w[0]), D(y[1], w[1]), D(y[2], w[2]), D(y[3], w[3]), D(y[4], w[4]), D(y[5], w[5]), D(y[6], w[6]), D(y[7], w[7]), e, g);
        dump("AN", e, g, 8);
    }
    return 0;
}
