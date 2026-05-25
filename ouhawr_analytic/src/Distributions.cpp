/**
 * @file   Distributions.cpp
 * @brief  Concrete duration distribution implementations.
 *         Gamma, Gaussian, Laplace (Eq.14-18), Poisson, Mixture (Eq.19-22).
 */

#include "MSDHMM.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ouhawr {

// ═══════════════════════════════════════════════════════════════════════════
//  Gamma
// ═══════════════════════════════════════════════════════════════════════════

double GammaDuration::logGamma(double x) {
    // Lanczos approximation
    if (x <= 0) return std::numeric_limits<double>::infinity();
    static const double g = 7.0;
    static const double c[] = {0.99999999999980993,676.5203681218851,-1259.1392167224028,
        771.32342877765313,-176.61502916214059,12.507343278686905,
        -0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7};
    if (x < 0.5) return std::log(M_PI/std::sin(M_PI*x)) - logGamma(1-x);
    x -= 1;
    double a = c[0];
    double t = x + g + 0.5;
    for (int i=1;i<=8;++i) a += c[i]/(x+i);
    return 0.5*std::log(2*M_PI) + (x+0.5)*std::log(t) - t + std::log(a);
}

double GammaDuration::digamma(double x) {
    // Asymptotic series
    if (x < 6) return digamma(x+1) - 1.0/x;
    double r = std::log(x) - 0.5/x;
    double xi2 = 1.0/(x*x);
    r -= xi2*(1.0/12 - xi2*(1.0/120 - xi2/252));
    return r;
}

double GammaDuration::prob(int d) const {
    if (d < 1) return 0.0;
    // Gamma PDF: p(d) = rate^shape / Γ(shape) * d^(shape-1) * exp(-rate*d)
    double logp = shape*std::log(rate) - logGamma(shape) +
                  (shape-1)*std::log((double)d) - rate*d;
    return std::exp(logp);
}

int GammaDuration::maxDuration() const {
    return (int)std::ceil(shape/rate + 5*std::sqrt(shape)/rate);
}

void GammaDuration::reestimate(const std::vector<double>& w) {
    // MLE for Gamma via Newton-Raphson on the log-likelihood.
    // w[i] is weight for duration i+1.
    double sw = 0, swlogd = 0, swd = 0;
    for (int i = 0; i < (int)w.size(); ++i) {
        double d = i + 1.0;
        sw += w[i]; swlogd += w[i]*std::log(d); swd += w[i]*d;
    }
    if (sw < 1e-15) return;
    double meanD    = swd / sw;
    double meanLogD = swlogd / sw;

    // Initial estimate via method of moments
    double logMean  = std::log(meanD);
    double s        = logMean - meanLogD; // s = log(μ) - E[log(D)]
    if (s <= 0) return;

    // Newton-Raphson
    double alpha = (3 - s + std::sqrt((s-3)*(s-3) + 24*s)) / (12*s);
    for (int it=0; it<20; ++it) {
        double g  = std::log(alpha) - digamma(alpha) - s;
        double gp = 1.0/alpha - 1.0; // d/dα [log α - ψ(α)]
        if (std::abs(gp) < 1e-15) break;
        double delta = -g / gp;
        alpha += delta;
        if (alpha < 1e-6) { alpha = 1e-6; break; }
        if (std::abs(delta) < 1e-8) break;
    }
    shape = alpha;
    rate  = alpha / meanD;
    if (rate < 1e-6) rate = 1e-6;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Gaussian
// ═══════════════════════════════════════════════════════════════════════════

double GaussianDuration::prob(int d) const {
    double z = (d - mu) / sigma;
    return std::exp(-0.5*z*z) / (sigma * std::sqrt(2*M_PI));
}

int GaussianDuration::maxDuration() const {
    return (int)std::ceil(mu + 4*sigma);
}

void GaussianDuration::reestimate(const std::vector<double>& w) {
    double sw = 0, swd = 0, swd2 = 0;
    for (int i = 0; i < (int)w.size(); ++i) {
        double d = i + 1.0;
        sw += w[i]; swd += w[i]*d; swd2 += w[i]*d*d;
    }
    if (sw < 1e-15) return;
    mu    = swd / sw;
    double var = swd2/sw - mu*mu;
    sigma = std::max(0.5, std::sqrt(std::max(var, 0.01)));
}

// ═══════════════════════════════════════════════════════════════════════════
//  Laplace  (§3.3.1, Eq. 14–18)
// ═══════════════════════════════════════════════════════════════════════════

double LaplaceDuration::prob(int d) const {
    // p_k(d) = (1/2ν) exp(-|d - μ| / ν)   [Eq. 14]
    return std::exp(-std::abs((double)d - mu) / nu) / (2.0 * nu);
}

int LaplaceDuration::maxDuration() const {
    return (int)std::ceil(mu + 6*nu);
}

void LaplaceDuration::reestimate(const std::vector<double>& w) {
    // ── Eq. (18): μ̄ via weighted-occupancy maximiser ──────────────────────
    // Approximated as the weighted median.
    double sw = 0;
    for (auto x : w) sw += x;
    if (sw < 1e-15) return;

    // Weighted median
    double half = sw / 2.0;
    double cum  = 0.0;
    double newMu = mu;
    for (int i = 0; i < (int)w.size(); ++i) {
        cum += w[i];
        if (cum >= half) { newMu = i + 1.0; break; }
    }
    mu = newMu;

    // ── Eq. (17): ν̄ = Σ_d |d − μ| w_d / Σ_d w_d ─────────────────────────
    double num = 0.0;
    for (int i = 0; i < (int)w.size(); ++i) {
        num += w[i] * std::abs((double)(i+1) - mu);
    }
    nu = std::max(0.01, num / sw);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Poisson
// ═══════════════════════════════════════════════════════════════════════════

double PoissonDuration::logFactorial(int n) {
    static std::vector<double> cache = {0.0};
    while ((int)cache.size() <= n)
        cache.push_back(cache.back() + std::log((double)cache.size()));
    return cache[n];
}

double PoissonDuration::prob(int d) const {
    if (d < 1) return 0.0;
    // p(d) = exp(-λ) λ^d / d!  [1-indexed: d ≥ 1]
    double logp = -lambda + d*std::log(lambda) - logFactorial(d);
    return std::exp(logp);
}

int PoissonDuration::maxDuration() const {
    return (int)std::ceil(lambda + 5*std::sqrt(lambda));
}

void PoissonDuration::reestimate(const std::vector<double>& w) {
    double sw = 0, swd = 0;
    for (int i = 0; i < (int)w.size(); ++i) { sw += w[i]; swd += w[i]*(i+1); }
    if (sw > 1e-15) lambda = std::max(0.1, swd/sw);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Mixture of Gamma + Laplace  (§3.3.2, Eq. 19–22)
// ═══════════════════════════════════════════════════════════════════════════

MixtureDuration::MixtureDuration() {
    components.push_back(std::make_unique<GammaDuration>(2.0, 0.4));
    components.push_back(std::make_unique<LaplaceDuration>(5.0, 2.0));
    coefficients = {0.5, 0.5};
}

MixtureDuration::MixtureDuration(const MixtureDuration& o)
    : coefficients(o.coefficients)
{
    for (auto& c : o.components) components.push_back(c->clone());
}

MixtureDuration& MixtureDuration::operator=(const MixtureDuration& o) {
    if (this != &o) {
        coefficients = o.coefficients;
        components.clear();
        for (auto& c : o.components) components.push_back(c->clone());
    }
    return *this;
}

double MixtureDuration::prob(int d) const {
    double p = 0.0;
    for (int m = 0; m < (int)components.size(); ++m)
        p += coefficients[m] * components[m]->prob(d);
    return p;
}

int MixtureDuration::maxDuration() const {
    int mx = 0;
    for (auto& c : components) mx = std::max(mx, c->maxDuration());
    return mx;
}

void MixtureDuration::reestimate(const std::vector<double>& w) {
    int M = (int)components.size();
    int D = (int)w.size();

    // Compute responsibilities r_m(d) = c_m * p^m(d) / Σ_m c_m * p^m(d)
    std::vector<std::vector<double>> r(M, std::vector<double>(D, 0.0));
    for (int d = 0; d < D; ++d) {
        double pSum = 0.0;
        for (int m = 0; m < M; ++m) pSum += coefficients[m] * components[m]->prob(d+1);
        if (pSum < 1e-300) continue;
        for (int m = 0; m < M; ++m)
            r[m][d] = coefficients[m] * components[m]->prob(d+1) / pSum;
    }

    // ── Eq. (21): Update mixture coefficients ─────────────────────────────
    double sw = std::accumulate(w.begin(), w.end(), 0.0);
    if (sw < 1e-15) return;

    std::vector<double> newCoeffs(M, 0.0);
    for (int m = 0; m < M; ++m) {
        for (int d = 0; d < D; ++d) newCoeffs[m] += r[m][d] * w[d];
        newCoeffs[m] /= sw;
    }
    // Normalize (Eq. 20: c_{k1} + c_{k2} = 1)
    double cSum = std::accumulate(newCoeffs.begin(), newCoeffs.end(), 0.0);
    if (cSum < 1e-15) return;
    for (auto& c : newCoeffs) c /= cSum;
    coefficients = newCoeffs;

    // ── Update each component's parameters with its weighted observations ──
    for (int m = 0; m < M; ++m) {
        std::vector<double> wm(D);
        for (int d = 0; d < D; ++d) wm[d] = r[m][d] * w[d];
        components[m]->reestimate(wm);
    }
}

std::unique_ptr<DurationDistribution> MixtureDuration::clone() const {
    return std::make_unique<MixtureDuration>(*this);
}

} // namespace ouhawr
