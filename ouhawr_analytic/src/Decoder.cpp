/**
 * @file   Decoder.cpp
 * @brief  TwoLevelDecoder (Algorithms 1 & 2) and StreamWeightOptimiser (Eq.25–26).
 */

#include "MSDHMM.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <unordered_set>

namespace ouhawr {

// ═══════════════════════════════════════════════════════════════════════════
//  Stream-weight optimisation  (§3.5, Eq.25–26)
// ═══════════════════════════════════════════════════════════════════════════

static std::vector<std::vector<double>>
getStreamObs(const ObsSequence& seq, int l) {
    std::vector<std::vector<double>> obs;
    obs.reserve(seq.size());
    for (auto& f : seq) obs.push_back(f.stream(l));
    return obs;
}

double StreamWeightOptimiser::mutualInfo(
        const StreamHMM& hmm,
        const std::vector<std::vector<double>>& obs)
{
    if (obs.empty()) return 0.0;
    int T = (int)obs.size();
    int N = hmm.N;

    std::vector<std::vector<double>> alpha, beta;
    double logProb = hmm.forwardBackward(obs, alpha, beta);
    if (logProb == LOG_ZERO) return 0.0;

    // Marginal state distribution P(q) via γ_t(k)
    std::vector<double> occ(N, 0.0);
    for (int t = 1; t <= T; ++t)
        for (int k = 0; k < N; ++k)
            if (alpha[t][k] != LOG_ZERO && beta[t][k] != LOG_ZERO)
                occ[k] += std::exp(alpha[t][k] + beta[t][k] - logProb);

    double totOcc = std::accumulate(occ.begin(), occ.end(), 0.0);
    if (totOcc < 1e-15) return 0.0;

    // H(Q) – marginal entropy
    double HQ = 0.0;
    for (int k = 0; k < N; ++k) {
        double p = occ[k] / totOcc;
        if (p > 1e-15) HQ -= p * std::log(p);
    }

    // H(Q|O) – conditional entropy (approximated via posterior at each time)
    double HQO = 0.0;
    for (int t = 1; t <= T; ++t) {
        double rowSum = 0.0;
        for (int k = 0; k < N; ++k)
            if (alpha[t][k] != LOG_ZERO && beta[t][k] != LOG_ZERO)
                rowSum += std::exp(alpha[t][k] + beta[t][k] - logProb);
        if (rowSum < 1e-15) continue;
        for (int k = 0; k < N; ++k) {
            if (alpha[t][k] == LOG_ZERO || beta[t][k] == LOG_ZERO) continue;
            double p = std::exp(alpha[t][k] + beta[t][k] - logProb) / rowSum;
            if (p > 1e-15) HQO -= p * std::log(p);
        }
    }
    HQO /= T;

    // I(O,Q) = H(Q) - H(Q|O)  [Eq. 26]
    return std::max(0.0, HQ - HQO);
}

double StreamWeightOptimiser::stateEntropy(const std::vector<double>& occ) {
    double tot = std::accumulate(occ.begin(), occ.end(), 0.0);
    double H   = 0.0;
    if (tot < 1e-15) return 0.0;
    for (double p : occ) { double q=p/tot; if(q>1e-15) H -= q*std::log(q); }
    return H;
}

std::array<double, NUM_STREAMS>
StreamWeightOptimiser::compute(
        const CharacterModel& charModel,
        const std::vector<ObsSequence>& trainObs)
{
    std::array<double, NUM_STREAMS> info = {};
    for (int l = 0; l < NUM_STREAMS; ++l) {
        double totalInfo = 0.0;
        for (auto& seq : trainObs) {
            auto obs = getStreamObs(seq, l);
            totalInfo += mutualInfo(charModel.streams[l], obs);
        }
        info[l] = totalInfo / std::max(1, (int)trainObs.size());
    }

    // Normalize to get weights  [Eq. 25]
    double sum = std::accumulate(info.begin(), info.end(), 0.0);
    std::array<double, NUM_STREAMS> weights;
    if (sum < 1e-15) {
        weights.fill(1.0 / NUM_STREAMS);
    } else {
        for (int l = 0; l < NUM_STREAMS; ++l) weights[l] = info[l] / sum;
    }
    return weights;
}

// ═══════════════════════════════════════════════════════════════════════════
//  TwoLevelDecoder
// ═══════════════════════════════════════════════════════════════════════════

TwoLevelDecoder::TwoLevelDecoder(
        const std::map<std::string, CharacterModel>& charModels,
        const std::map<std::string, std::array<double, NUM_STREAMS>>& weights)
    : models_(charModels), weights_(weights)
{}

// ── Level-1 precomputation ────────────────────────────────────────────────

TwoLevelDecoder::Level1Cache
TwoLevelDecoder::runLevel1(
        const ObsSequence& obs,
        const std::vector<std::string>& candidates,
        const std::map<std::string, std::vector<std::string>>& lexicon) const
{
    Level1Cache cache;
    int T = (int)obs.size();
    cache.T = T;

    // Collect unique characters needed from candidates
    std::unordered_set<std::string> needed;
    for (auto& w : candidates) {
        auto it = lexicon.find(w);
        if (it == lexicon.end()) continue;
        for (auto& c : it->second) needed.insert(c);
    }

    // Algorithm 1: for each character model c, for each stream l
    for (auto& charLabel : needed) {
        auto mit = models_.find(charLabel);
        if (mit == models_.end()) continue;
        const CharacterModel& cm = mit->second;

        for (int l = 0; l < NUM_STREAMS; ++l) {
            auto streamObs = getStreamObs(obs, l);
            // chi[s][e] and epsilon[s][e]
            std::vector<std::vector<double>>              chi;
            std::vector<std::vector<std::vector<int>>>    eps;
            cm.streams[l].level1Decode(streamObs, chi, eps);
            cache.chi[charLabel][l]     = chi;
            cache.epsilon[charLabel][l] = eps;
        }
    }
    return cache;
}

// ── Combine per-stream scores (weighted sum) ──────────────────────────────

std::vector<std::vector<double>>
TwoLevelDecoder::combineStreams(
        const std::string& charLabel,
        const std::array<std::vector<std::vector<double>>, NUM_STREAMS>& chi,
        int T) const
{
    // χ^c(s,e) = Σ_l w^l_c * χ^{cl}(s,e)   [Algorithm 2, line 3]
    std::vector<std::vector<double>> combined(T, std::vector<double>(T, LOG_ZERO));

    // Get stream weights for this character
    std::array<double, NUM_STREAMS> w;
    w.fill(1.0 / NUM_STREAMS);
    auto wit = weights_.find(charLabel);
    if (wit != weights_.end()) w = wit->second;

    for (int s = 0; s < T; ++s) {
        for (int e = s+1; e < T; ++e) {
            double score = 0.0;
            for (int l = 0; l < NUM_STREAMS; ++l) {
                if ((int)chi[l].size() <= s || (int)chi[l][s].size() <= e) continue;
                double cl = chi[l][s][e];
                if (cl == LOG_ZERO) continue;
                score += w[l] * cl;
            }
            combined[s][e] = (score == 0.0) ? LOG_ZERO : score;
        }
    }
    return combined;
}

// ── Algorithm 2: decode one word ─────────────────────────────────────────

std::pair<double, std::vector<int>>
TwoLevelDecoder::runLevel2Word(
        const std::vector<std::string>& charSeq,
        const std::map<std::string, std::vector<std::vector<double>>>& combinedChi,
        int T) const
{
    int K = (int)charSeq.size();
    if (K == 0) return {LOG_ZERO, {}};

    // δ̂_t(k): best log-prob ending at character k at time t  [Algorithm 2]
    std::vector<std::vector<double>> deltaHat(T, std::vector<double>(K, LOG_ZERO));
    std::vector<std::vector<int>>    omegaHat(T, std::vector<int>(K, -1));

    // Step 2: Initialisation  [Algorithm 2, line 6-8]
    for (int t = 1; t < T; ++t) {
        // k=0: δ̂_t(0) = χ^{c_1}(1,t)
        auto it0 = combinedChi.find(charSeq[0]);
        if (it0 != combinedChi.end() && t < (int)it0->second.size())
            deltaHat[t][0] = it0->second[0][t];
    }

    for (int k = 1; k < K; ++k) {
        auto it = combinedChi.find(charSeq[k]);
        if (it == combinedChi.end()) continue;
        for (int t = k+1; t < T; ++t) {
            double best = LOG_ZERO; int bestS = -1;
            for (int s = k; s < t; ++s) {
                if (deltaHat[s][k-1] == LOG_ZERO) continue;
                double chi_k = (s+1 < (int)it->second.size() && t < (int)it->second[s+1].size())
                               ? it->second[s+1][t] : LOG_ZERO;
                if (chi_k == LOG_ZERO) continue;
                double score = deltaHat[s][k-1] + chi_k;
                if (score > best) { best = score; bestS = s; }
            }
            deltaHat[t][k] = best;
            omegaHat[t][k] = bestS;
        }
    }

    // Step 3: Termination  [Algorithm 2, line 10-11]
    double bestScore = deltaHat[T-1][K-1];

    // Step 4: Character backtracking  [Algorithm 2, line 13-14]
    std::vector<int> charBoundaries(K, -1);
    int t = T-1;
    for (int k = K-1; k >= 0; --k) {
        charBoundaries[k] = t;
        if (k > 0 && omegaHat[t][k] >= 0) t = omegaHat[t][k];
    }

    return {bestScore, charBoundaries};
}

// ── Main decode entry point ────────────────────────────────────────────────

DecodeResult TwoLevelDecoder::decode(
        const ObsSequence& obs,
        const std::vector<std::string>& nBest,
        const std::map<std::string, std::vector<std::string>>& lexicon) const
{
    if (obs.empty() || nBest.empty())
        return {"", LOG_ZERO, {}, {}};

    int T = (int)obs.size();

    // ── Level 1 ──────────────────────────────────────────────────────────
    Level1Cache l1 = runLevel1(obs, nBest, lexicon);

    // Build combined per-character score matrices
    std::map<std::string, std::vector<std::vector<double>>> combinedChi;
    for (auto& [charLabel, chiArr] : l1.chi) {
        combinedChi[charLabel] = combineStreams(charLabel, chiArr, T);
    }

    // ── Level 2 ──────────────────────────────────────────────────────────
    double bestScore = LOG_ZERO;
    std::string bestWord;
    std::vector<int> bestBoundaries;

    for (auto& word : nBest) {
        auto lit = lexicon.find(word);
        if (lit == lexicon.end()) continue;
        auto [score, boundaries] = runLevel2Word(lit->second, combinedChi, T);
        if (score > bestScore) {
            bestScore      = score;
            bestWord       = word;
            bestBoundaries = boundaries;
        }
    }

    // Build result
    DecodeResult res;
    res.word           = bestWord;
    res.logLikelihood  = bestScore;
    if (bestWord.empty()) return res;

    auto lit = lexicon.find(bestWord);
    if (lit != lexicon.end()) res.charSeq = lit->second;

    // Recover per-stream state paths from ε
    int K = (int)res.charSeq.size();
    res.statePaths.resize(K);
    for (int k = 0; k < K; ++k) {
        const std::string& cl = res.charSeq[k];
        int s = (k == 0) ? 0 : (bestBoundaries[k-1]+1);
        int e = bestBoundaries[k];
        if (s > e || e >= T) continue;
        auto eit = l1.epsilon.find(cl);
        if (eit == l1.epsilon.end()) continue;
        for (int l = 0; l < NUM_STREAMS; ++l) {
            if (s < (int)eit->second[l].size() && e < (int)eit->second[l][s].size())
                res.statePaths[k][l] = eit->second[l][s][e];
        }
    }

    return res;
}

} // namespace ouhawr
