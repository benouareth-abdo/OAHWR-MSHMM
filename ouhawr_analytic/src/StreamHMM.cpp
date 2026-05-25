/**
 * @file   StreamHMM.cpp
 * @brief  Single-stream CHSMM: Gaussian emission, forward-backward (Eq.8-12),
 *         Viterbi with explicit duration, and level-1 decode (Algorithm 1).
 */

#include "MSDHMM.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>

namespace ouhawr {

// ═══════════════════════════════════════════════════════════════════════════
//  GaussComp
// ═══════════════════════════════════════════════════════════════════════════

GaussComp::GaussComp(int dim, double initVar)
    : mean(dim, 0.0), var(dim, initVar), weight(1.0) {}

double GaussComp::logLikelihood(const std::vector<double>& obs) const {
    assert((int)obs.size() == (int)mean.size());
    double ll = 0.0;
    int d = (int)mean.size();
    for (int i = 0; i < d; ++i) {
        double v = std::max(var[i], 1e-6);
        double diff = obs[i] - mean[i];
        ll -= 0.5 * (std::log(2*M_PI*v) + diff*diff/v);
    }
    return ll;
}

// ═══════════════════════════════════════════════════════════════════════════
//  GMMEmission
// ═══════════════════════════════════════════════════════════════════════════

GMMEmission::GMMEmission(int d, int M) : dim(d) {
    for (int m = 0; m < M; ++m) {
        GaussComp gc(d, 1.0);
        gc.weight = 1.0 / M;
        components.push_back(std::move(gc));
    }
}

double GMMEmission::logLikelihood(const std::vector<double>& obs) const {
    double p = 0.0;
    for (auto& gc : components)
        p += gc.weight * std::exp(gc.logLikelihood(obs));
    return safeLog(p);
}

// ═══════════════════════════════════════════════════════════════════════════
//  StreamHMM construction
// ═══════════════════════════════════════════════════════════════════════════

static std::unique_ptr<DurationDistribution> makeDur(const std::string& type) {
    if (type == "Gamma")    return std::make_unique<GammaDuration>();
    if (type == "Gaussian") return std::make_unique<GaussianDuration>();
    if (type == "Laplace")  return std::make_unique<LaplaceDuration>();
    if (type == "Poisson")  return std::make_unique<PoissonDuration>();
    if (type == "Mixture")  return std::make_unique<MixtureDuration>();
    throw std::invalid_argument("Unknown duration type: " + type);
}

StreamHMM::StreamHMM(int states, int gaussMix, int featureDim,
                     const std::string& durType)
    : N(states), M(gaussMix), dim(featureDim)
{
    pi.assign(N, 1.0/N);
    A.assign(N, std::vector<double>(N, 0.0));

    // Right-to-left topology: allow self (skipped), +1, +2 (skip one state)
    for (int i = 0; i < N-1; ++i) {
        A[i][i+1] = 0.7;
        if (i < N-2) A[i][i+2] = 0.3;
        // Normalize
        double s = A[i][i+1] + (i<N-2 ? A[i][i+2] : 0.0);
        A[i][i+1] /= s;
        if (i < N-2) A[i][i+2] /= s;
    }
    A[N-1][N-1] = 1.0;

    for (int i = 0; i < N; ++i)
        B.push_back(GMMEmission(dim, M));

    initDurations(durType);
}

void StreamHMM::initDurations(const std::string& durType) {
    dur.clear();
    for (int i = 0; i < N; ++i) dur.push_back(makeDur(durType));
}

StreamHMM::StreamHMM(const StreamHMM& o)
    : N(o.N), M(o.M), dim(o.dim), pi(o.pi), A(o.A), B(o.B)
{
    for (auto& d : o.dur) dur.push_back(d->clone());
}

StreamHMM& StreamHMM::operator=(const StreamHMM& o) {
    if (this != &o) {
        N=o.N; M=o.M; dim=o.dim; pi=o.pi; A=o.A; B=o.B;
        dur.clear();
        for (auto& d : o.dur) dur.push_back(d->clone());
    }
    return *this;
}

void StreamHMM::randomInit(std::mt19937& rng) {
    std::normal_distribution<> nd(0.0, 0.5);
    std::uniform_real_distribution<> ud(0.0,1.0);
    for (int i = 0; i < N; ++i) {
        for (auto& gc : B[i].components) {
            for (auto& m : gc.mean) m = nd(rng);
            for (auto& v : gc.var)  v = 0.5 + ud(rng);
        }
    }
}

void StreamHMM::normalizeA() {
    for (int i = 0; i < N; ++i) {
        double s = 0;
        for (int j = 0; j < N; ++j) s += A[i][j];
        if (s > 1e-15) for (int j = 0; j < N; ++j) A[i][j] /= s;
    }
}

double StreamHMM::emissionLogProb(int state, const std::vector<double>& o) const {
    return B[state].logLikelihood(o);
}

int StreamHMM::maxDur() const {
    int mx = 1;
    for (auto& d : dur) mx = std::max(mx, d->maxDuration());
    return std::min(mx, 30); // cap for efficiency
}

// ═══════════════════════════════════════════════════════════════════════════
//  Forward-backward with explicit duration  (Eq. 8–12)
// ═══════════════════════════════════════════════════════════════════════════

double StreamHMM::forwardBackward(
        const std::vector<std::vector<double>>& obs,
        std::vector<std::vector<double>>& alpha,
        std::vector<std::vector<double>>& beta) const
{
    int T = (int)obs.size();
    int D = maxDur();

    // alpha[t][k] = log P(o_1..o_t, q_t = k)  [Eq. 10]
    alpha.assign(T+1, std::vector<double>(N, LOG_ZERO));
    beta.assign (T+1, std::vector<double>(N, LOG_ZERO));

    // Initialisation (Eq. 9): α_0(k) = π_k
    for (int k = 0; k < N; ++k) alpha[0][k] = safeLog(pi[k]);

    // Forward recursion (Eq. 10)
    for (int t = 1; t <= T; ++t) {
        for (int k = 0; k < N; ++k) {
            double sumAlpha = LOG_ZERO;
            for (int d = 1; d <= std::min(t, D); ++d) {
                // Compute Π_{s=t-d+1}^{t} b_k(o_s)
                double emitSum = 0.0;
                for (int s = t-d+1; s <= t; ++s)
                    emitSum += emissionLogProb(k, obs[s-1]);

                double durLog = dur[k]->logProb(d);
                if (durLog == LOG_ZERO) continue;

                // Sum over previous states i ≠ k
                double prevSum = LOG_ZERO;
                for (int i = 0; i < N; ++i) {
                    if (i == k) continue;
                    double aLog = safeLog(A[i][k]);
                    if (aLog == LOG_ZERO || alpha[t-d][i] == LOG_ZERO) continue;
                    prevSum = logAdd(prevSum, alpha[t-d][i] + aLog);
                }
                if (prevSum == LOG_ZERO) continue;
                sumAlpha = logAdd(sumAlpha, prevSum + durLog + emitSum);
            }
            alpha[t][k] = sumAlpha;
        }
    }

    // Backward recursion (Eq. 11–12)
    for (int i = 0; i < N; ++i) beta[T][i] = LOG_ONE;

    for (int t = T-1; t >= 0; --t) {
        for (int i = 0; i < N; ++i) {
            double sumBeta = LOG_ZERO;
            for (int k = 0; k < N; ++k) {
                if (k == i) continue;
                double aLog = safeLog(A[i][k]);
                if (aLog == LOG_ZERO) continue;
                for (int d = 1; d <= std::min(T-t, D); ++d) {
                    double durLog = dur[k]->logProb(d);
                    if (durLog == LOG_ZERO) continue;
                    double emitSum = 0.0;
                    for (int s = t+1; s <= t+d && s <= T; ++s)
                        emitSum += emissionLogProb(k, obs[s-1]);
                    if (t+d > T) continue;
                    sumBeta = logAdd(sumBeta,
                        aLog + durLog + emitSum + beta[t+d][k]);
                }
            }
            beta[t][i] = sumBeta;
        }
    }

    // log P(O|λ) = log Σ_k α_T(k)
    double logProb = LOG_ZERO;
    for (int k = 0; k < N; ++k) logProb = logAdd(logProb, alpha[T][k]);
    return logProb;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Viterbi with explicit duration
// ═══════════════════════════════════════════════════════════════════════════

double StreamHMM::viterbi(const std::vector<std::vector<double>>& obs,
                           std::vector<int>& bestPath,
                           std::vector<int>& bestDurs) const
{
    int T = (int)obs.size();
    int D = maxDur();

    // delta[t][k] = best log-prob of path ending in state k at time t
    std::vector<std::vector<double>> delta(T+1, std::vector<double>(N, LOG_ZERO));
    // psi[t][k] = {previous state, duration}
    std::vector<std::vector<std::pair<int,int>>> psi(T+1,
        std::vector<std::pair<int,int>>(N, {-1,0}));

    for (int k = 0; k < N; ++k) delta[0][k] = safeLog(pi[k]);

    for (int t = 1; t <= T; ++t) {
        for (int k = 0; k < N; ++k) {
            double best = LOG_ZERO;
            int bState=-1, bDur=0;
            for (int d = 1; d <= std::min(t,D); ++d) {
                double durLog = dur[k]->logProb(d);
                if (durLog == LOG_ZERO) continue;
                double emitSum = 0.0;
                for (int s = t-d+1; s <= t; ++s)
                    emitSum += emissionLogProb(k, obs[s-1]);
                for (int i = 0; i < N; ++i) {
                    if (i == k) continue;
                    double aLog = safeLog(A[i][k]);
                    if (delta[t-d][i] == LOG_ZERO) continue;
                    double score = delta[t-d][i] + aLog + durLog + emitSum;
                    if (score > best) { best=score; bState=i; bDur=d; }
                }
            }
            delta[t][k] = best;
            psi[t][k] = {bState, bDur};
        }
    }

    // Termination
    double bestScore = LOG_ZERO;
    int lastState = 0;
    for (int k = 0; k < N; ++k)
        if (delta[T][k] > bestScore) { bestScore = delta[T][k]; lastState = k; }

    // Backtrack
    bestPath.clear(); bestDurs.clear();
    int t = T, s = lastState;
    while (t > 0 && s >= 0) {
        auto [prevS, dur_d] = psi[t][s];
        for (int i = 0; i < dur_d; ++i) bestPath.push_back(s);
        bestDurs.push_back(dur_d);
        t -= dur_d; s = prevS;
    }
    std::reverse(bestPath.begin(), bestPath.end());
    std::reverse(bestDurs.begin(), bestDurs.end());
    return bestScore;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Level-1 decode (Algorithm 1)
// ═══════════════════════════════════════════════════════════════════════════

void StreamHMM::level1Decode(
        const std::vector<std::vector<double>>& obs,
        std::vector<std::vector<double>>& chi,
        std::vector<std::vector<std::vector<int>>>& epsilon) const
{
    int T = (int)obs.size();

    chi.assign(T, std::vector<double>(T, LOG_ZERO));
    epsilon.assign(T, std::vector<std::vector<int>>(T));

    // Algorithm 1 – for each starting frame s ∈ {0..T-1}
    for (int s = 0; s < T; ++s) {
        // δ_t(i): best log-prob of reaching state i exactly at time t,
        //         with the character starting at frame s.
        // psi_t(i) = {previous state, duration d that brought us to (t,i)}
        std::vector<std::vector<double>> delta(T, std::vector<double>(N, LOG_ZERO));
        std::vector<std::vector<std::pair<int,int>>> psi(T,
            std::vector<std::pair<int,int>>(N, {-1,0}));

        // ── Step 1: Initialisation (Algorithm 1, lines 7–8) ──────────────
        // δ_s(0) = π_0 · p_0(1) · b_0(O_s)
        {
            double logPi  = safeLog(pi[0]);
            double logDur = dur[0]->logProb(1);
            double logB   = emissionLogProb(0, obs[s]);
            delta[s][0] = logPi + logDur + logB;
            psi[s][0] = {-1, 1};
        }

        // ── Step 2: Recursion (Algorithm 1, lines 9–15) ──────────────────
        for (int t = s+1; t < T; ++t) {
            // State 0: spans [s, t] with duration d = t-s+1 directly from π
            {
                double durLog = dur[0]->logProb(t - s + 1);
                if (durLog != LOG_ZERO) {
                    double emit = 0.0;
                    for (int k = s; k <= t; ++k) emit += emissionLogProb(0, obs[k]);
                    delta[t][0] = safeLog(pi[0]) + durLog + emit;
                    psi[t][0] = {-1, t-s+1};
                }
            }

            for (int i = 1; i < N; ++i) {
                double best = LOG_ZERO;
                int bState = -1, bDur = 0;
                // Previous end-time tau ∈ {s, …, t-1}
                for (int tau = s; tau < t; ++tau) {
                    int d = t - tau;
                    double durLog = dur[i]->logProb(d);
                    if (durLog == LOG_ZERO) continue;
                    // Emission product: b_i(O_{tau+1}) … b_i(O_t)
                    double emit = 0.0;
                    for (int k = tau+1; k <= t; ++k) emit += emissionLogProb(i, obs[k]);
                    // Best predecessor state j ≠ i ending at tau
                    for (int j = 0; j < N; ++j) {
                        if (j == i) continue;
                        double aLog = safeLog(A[j][i]);
                        if (aLog == LOG_ZERO || delta[tau][j] == LOG_ZERO) continue;
                        double score = delta[tau][j] + aLog + durLog + emit;
                        if (score > best) { best = score; bState = j; bDur = d; }
                    }
                }
                delta[t][i] = best;
                psi[t][i]   = {bState, bDur};
            }
        }

        // ── Step 3 & 4: Termination + path backtracking (lines 17–25) ────
        for (int e = s+1; e < T; ++e) {
            chi[s][e] = delta[e][N-1];   // χ(s,e) = δ_e(N-1)

            // Backtrack optimal state path for this segment
            std::vector<int> path(T, -1);
            int t2 = e, st = N-1;
            while (t2 >= s && st >= 0) {
                auto [prevSt, d] = psi[t2][st];
                int kStart = std::max(s, t2 - d + 1);
                for (int k = kStart; k <= t2 && k < T; ++k) path[k] = st;
                t2 -= d; st = prevSt;
            }
            epsilon[s][e] = path;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Baum-Welch step
// ═══════════════════════════════════════════════════════════════════════════

double StreamHMM::baumWelchStep(const std::vector<std::vector<double>>& obs) {
    int T = (int)obs.size();
    int D = maxDur();

    std::vector<std::vector<double>> alpha, beta;
    double logProb = forwardBackward(obs, alpha, beta);
    if (logProb == LOG_ZERO) return logProb;

    // Compute state occupation γ_t(k) = α_t(k) + β_t(k) - logProb
    std::vector<std::vector<double>> gamma(T+1, std::vector<double>(N, LOG_ZERO));
    for (int t = 0; t <= T; ++t)
        for (int k = 0; k < N; ++k)
            if (alpha[t][k] != LOG_ZERO && beta[t][k] != LOG_ZERO)
                gamma[t][k] = alpha[t][k] + beta[t][k] - logProb;

    // Duration occupation: for each state k, accumulate weighted duration counts
    std::vector<std::vector<double>> durWeights(N, std::vector<double>(D+1, 0.0));
    for (int k = 0; k < N; ++k) {
        for (int t = 1; t <= T; ++t) {
            for (int d = 1; d <= std::min(t,D); ++d) {
                double durLog = dur[k]->logProb(d);
                double emit = 0;
                for (int s=t-d+1;s<=t;++s) emit += emissionLogProb(k, obs[s-1]);
                double prevSum = LOG_ZERO;
                for (int i=0;i<N;++i) {
                    if(i==k) continue;
                    double a=safeLog(A[i][k]);
                    if(alpha[t-d][i]==LOG_ZERO||a==LOG_ZERO) continue;
                    prevSum=logAdd(prevSum, alpha[t-d][i]+a);
                }
                if(prevSum==LOG_ZERO||durLog==LOG_ZERO||beta[t][k]==LOG_ZERO) continue;
                double xi = prevSum + durLog + emit + beta[t][k] - logProb;
                if (d-1 < (int)durWeights[k].size())
                    durWeights[k][d-1] += std::exp(xi);
            }
        }
        // Re-estimate duration distribution (Eq. 17–18)
        dur[k]->reestimate(durWeights[k]);
    }

    // GMM emission re-estimation
    for (int k = 0; k < N; ++k) {
        int nmix = (int)B[k].components.size();
        std::vector<double> accW(nmix, 0.0);
        std::vector<std::vector<double>> accMean(nmix, std::vector<double>(dim,0.0));
        std::vector<std::vector<double>> accVar(nmix, std::vector<double>(dim,0.0));

        for (int t = 1; t <= T; ++t) {
            if (gamma[t][k] == LOG_ZERO) continue;
            double gkt = std::exp(gamma[t][k]);
            for (int m = 0; m < nmix; ++m) {
                auto& gc = B[k].components[m];
                double compLL = gc.weight * std::exp(gc.logLikelihood(obs[t-1]));
                double totLL  = std::exp(B[k].logLikelihood(obs[t-1]));
                double resp   = (totLL > 1e-300) ? gkt * compLL / totLL : 0.0;
                accW[m] += resp;
                for (int d2=0;d2<dim;++d2) {
                    accMean[m][d2] += resp * obs[t-1][d2];
                    accVar[m][d2]  += resp * obs[t-1][d2]*obs[t-1][d2];
                }
            }
        }
        // Update GMM parameters
        double totW = std::accumulate(accW.begin(),accW.end(),0.0);
        if (totW < 1e-15) continue;
        for (int m = 0; m < nmix; ++m) {
            if (accW[m] < 1e-15) continue;
            B[k].components[m].weight = accW[m]/totW;
            for (int d2=0;d2<dim;++d2) {
                double mu_new = accMean[m][d2]/accW[m];
                B[k].components[m].mean[d2] = mu_new;
                B[k].components[m].var[d2]  =
                    std::max(0.01, accVar[m][d2]/accW[m] - mu_new*mu_new);
            }
        }
    }

    // Re-estimate transition matrix A
    // (simplified: use gamma ratios)
    for (int i = 0; i < N; ++i) {
        double denom = LOG_ZERO;
        for (int t=0;t<T;++t) denom = logAdd(denom, gamma[t][i]);
        for (int j = 0; j < N; ++j) {
            if (A[i][j] == 0.0) continue;
            double num = LOG_ZERO;
            for (int t=1;t<=T;++t) {
                double xi = alpha[t-1][i] + safeLog(A[i][j]) +
                            emissionLogProb(j,obs[t-1]) + beta[t][j] - logProb;
                num = logAdd(num, xi);
            }
            A[i][j] = (denom==LOG_ZERO) ? 0.0 : std::exp(num-denom);
        }
    }
    normalizeA();
    return logProb;
}

double StreamHMM::viterbiTrainStep(const std::vector<std::vector<double>>& obs) {
    std::vector<int> path, durs;
    double score = viterbi(obs, path, durs);
    // Use the Viterbi path to accumulate sufficient statistics
    // (simplified embedded training step)
    int T = (int)obs.size();
    if ((int)path.size() != T) return score;

    // Duration statistics from Viterbi path
    std::vector<std::vector<double>> durW(N);
    for (auto& dw : durW) dw.assign(30, 0.0);

    int di = 0;
    for (int d : durs) {
        int st = path[di];
        if (d > 0 && d <= 30) durW[st][d-1] += 1.0;
        di += d;
    }
    for (int k = 0; k < N; ++k) dur[k]->reestimate(durW[k]);

    // GMM update from Viterbi assignment
    std::vector<double> stateCount(N, 0.0);
    for (int t=0;t<T;++t) stateCount[path[t]] += 1.0;
    for (int k=0;k<N;++k) {
        int nmix = (int)B[k].components.size();
        std::vector<double> accW(nmix, 0.0);
        std::vector<std::vector<double>> accM(nmix, std::vector<double>(dim,0.0));
        std::vector<std::vector<double>> accV(nmix, std::vector<double>(dim,0.0));
        for (int t=0;t<T;++t) {
            if (path[t] != k) continue;
            double totLL = std::exp(B[k].logLikelihood(obs[t]));
            for (int m=0;m<nmix;++m) {
                double r = (totLL>1e-300) ?
                    B[k].components[m].weight*std::exp(B[k].components[m].logLikelihood(obs[t]))/totLL : 0.0;
                accW[m]+=r;
                for(int d2=0;d2<dim;++d2){
                    accM[m][d2]+=r*obs[t][d2]; accV[m][d2]+=r*obs[t][d2]*obs[t][d2];
                }
            }
        }
        double tw=std::accumulate(accW.begin(),accW.end(),0.0);
        if(tw<1e-15) continue;
        for(int m=0;m<nmix;++m){
            if(accW[m]<1e-15) continue;
            B[k].components[m].weight=accW[m]/tw;
            for(int d2=0;d2<dim;++d2){
                double mu=accM[m][d2]/accW[m];
                B[k].components[m].mean[d2]=mu;
                B[k].components[m].var[d2]=std::max(0.01,accV[m][d2]/accW[m]-mu*mu);
            }
        }
    }
    return score;
}

// ═══════════════════════════════════════════════════════════════════════════
//  CharacterModel
// ═══════════════════════════════════════════════════════════════════════════

CharacterModel::CharacterModel(const std::string& lbl,
                               const std::array<int, NUM_STREAMS>& dims,
                               int states, int mix,
                               const std::string& durType)
    : label(lbl)
{
    for (int l = 0; l < NUM_STREAMS; ++l)
        streams[l] = StreamHMM(states, mix, dims[l], durType);
}

} // namespace ouhawr
