#include "sdot_w2.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

struct Dirac {
    float x;
    float w;
    size_t original_index;
};

void sdot_w2_cpu_single(const float* Xf, const float* Wf, size_t Nf,
                        const float* Xg, const float* Yg, size_t Mg,
                        float* result, float* barycenters) {
    if (Nf == 0 || Mg < 2) {
        if (result) *result = 0.0f;
        if (barycenters) {
            for (size_t i = 0; i < Nf; ++i) barycenters[i] = 0.0f;
        }
        return;
    }

    // 1. Sort Diracs
    std::vector<Dirac> diracs(Nf);
    for (size_t i = 0; i < Nf; ++i) {
        diracs[i] = {Xf[i], Wf[i], i};
    }
    std::sort(diracs.begin(), diracs.end(), [](const Dirac& a, const Dirac& b) {
        return a.x < b.x;
    });

    // 2. Cumulative Mass of g. We add an additional "fake" value to handle precision issues
    std::vector<double> Cg(Mg + 1, 0.0);
    for (size_t j = 0; j < Mg - 1; ++j) {
        double dx = Xg[j+1] - Xg[j];
        double avg_y = 0.5 * (Yg[j] + Yg[j+1]);
        Cg[j+1] = Cg[j] + avg_y * dx;
    }
    Cg[Mg] = Cg[Mg - 1] * 10.0;

    // Normalize weights: scale f to match g's total mass
    double total_wf = 0.0;
    for (size_t i = 0; i < Nf; ++i)
        total_wf += diracs[i].w;

    double total_cg = Cg[Mg - 1];
    double scale_f = (total_wf > 0) ? (total_cg / total_wf) : 1.0;

    // Scale Dirac weights once
    // for (size_t i = 0; i < Nf; ++i) {
    //     diracs[i].w *= ;
    // }

    // 3. Find matching intervals and barycenters
    double current_mass_f = 0.0;
    double x_prev = Xg[0];
    double sum_sq_dist = 0.0;
    size_t g_idx = 0;

    for (size_t i = 0; i < Nf; ++i) {
        double mass_to_take = diracs[i].w * scale_f;
        double target_mass = current_mass_f + mass_to_take;

        // Use the scaled mass directly to find the interval in g
        while (g_idx < Mg - 1 && Cg[g_idx + 1] < target_mass - 1e-9) {
            g_idx++;
        }

        double m_needed = target_mass - Cg[g_idx];
        double x0 = Xg[g_idx];
        double x1 = Xg[g_idx+1];
        double y0 = Yg[g_idx];
        double y1 = Yg[g_idx+1];
        double h = x1 - x0;
        double dy = y1 - y0;

        double u;
        if (std::abs(dy) < 1e-9) {
            u = (y0 > 0) ? (m_needed / y0) : 0.0;
        } else {
            // Solve (dy/(2h))u^2 + y0*u - m_needed = 0
            double a = dy / (2.0 * h);
            double b = y0;
            double c = -m_needed;
            double delta = b * b - 4.0 * a * c;
            u = (-b + std::sqrt(std::max(0.0, delta))) / (2.0 * a);
        }
        double x_curr = x0 + u;

        auto calc_moment = [&](double a, double b, double X0, double X1, double Y0, double Y1) {
            double H = X1 - X0;
            double DY = Y1 - Y0;
            double slope = DY / H;
            auto integral = [&](double t) {
                return Y0 * t * t / 2.0 + slope * (t * t * t / 3.0 - X0 * t * t / 2.0);
            };
            return integral(b) - integral(a);
        };

        size_t start_g = 0;
        while (start_g < Mg - 1 && Xg[start_g + 1] < x_prev + 1e-9)
            start_g++;

        double total_moment = 0.0;
        for (size_t j = start_g; j <= g_idx; ++j) {
            double a = std::max(x_prev, (double)Xg[j]);
            double b = std::min(x_curr, (double)Xg[j+1]);
            if (a < b) {
                total_moment += calc_moment(a, b, Xg[j], Xg[j+1], Yg[j], Yg[j+1]);
            }
        }

        // Barycenter is total_moment / mass_of_interval
        // By construction, mass_of_interval is diracs[i].w (scaled)
        double barycenter = ( mass_to_take > 0 ) ? ( total_moment / mass_to_take ) : diracs[i].x;

        if ( barycenters ) {
            barycenters[diracs[i].original_index] = (float)barycenter;
        }

        double diff = diracs[i].x - barycenter;
        sum_sq_dist += diff * diff;

        x_prev = x_curr;
        current_mass_f = target_mass;
    }

    if (result) *result = (float)sum_sq_dist;
}

void sdot_w2_cpu(const float* Xf, const float* Wf, size_t Nf,
                 const float* Xg, const float* Yg, size_t Mg,
                 size_t batch_size,
                 float* result, float* barycenters) {
    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        sdot_w2_cpu_single(Xf + b * Nf, Wf + b * Nf, Nf,
                           Xg + b * Mg, Yg + b * Mg, Mg,
                           result + b,
                           barycenters ? (barycenters + b * Nf) : nullptr);
    }
}
