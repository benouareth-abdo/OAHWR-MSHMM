/**
 * @file   EmbeddedTrainer.cpp
 * @brief  EmbeddedTrainer: trains character-level CHSMMs from word-level samples.
 *         Also implements MSDHMM top-level train/recognize/save/load.
 *
 * §5.3 of Benouareth et al. (Information Fusion, 2026).
 */

#include "MSDHMM.hpp"
#include "Preprocessing.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <set>
#include <random>

namespace ouhawr {

// ═══════════════════════════════════════════════════════════════════════════
//  Helper
// ═══════════════════════════════════════════════════════════════════════════

std::vector<std::vector<double>>
EmbeddedTrainer::getStreamObs(const ObsSequence& seq, int l) {
    std::vector<std::vector<double>> obs;
    obs.reserve(seq.size());
    for (auto& f : seq) obs.push_back(f.stream(l));
    return obs;
}

// ═══════════════════════════════════════════════════════════════════════════
//  One Baum-Welch EM step over entire corpus  (embedded training §5.3)
// ═══════════════════════════════════════════════════════════════════════════

double EmbeddedTrainer::embeddedBWStep(
        std::map<std::string, CharacterModel>& models,
        const std::map<std::string, std::vector<ObsSequence>>& corpus,
        const std::map<std::string, std::vector<std::string>>& lexicon) const
{
    double totalLL = 0.0;
    int seqCount   = 0;

    for (auto& [wordLabel, seqs] : corpus) {
        auto lit = lexicon.find(wordLabel);
        if (lit == lexicon.end()) continue;
        const auto& charSeq = lit->second;

        for (auto& seq : seqs) {
            if (seq.empty()) continue;

            // For each stream, train each character HMM independently
            // (embedded: whole-word sequence, character boundaries found implicitly)
            for (int l = 0; l < NUM_STREAMS; ++l) {
                auto obs = getStreamObs(seq, l);
                if (obs.empty()) continue;

                // Concatenate character models for this word (embedded training)
                // We apply Baum-Welch to each character model segment
                // Simple approach: divide obs evenly among characters
                int T   = (int)obs.size();
                int K   = (int)charSeq.size();
                if (K == 0) continue;

                int segLen = std::max(1, T / K);
                for (int k = 0; k < K; ++k) {
                    auto it = models.find(charSeq[k]);
                    if (it == models.end()) continue;

                    int tStart = k * segLen;
                    int tEnd   = (k == K-1) ? T : std::min(T, (k+1) * segLen);
                    if (tStart >= tEnd) continue;

                    std::vector<std::vector<double>> segObs(
                        obs.begin() + tStart, obs.begin() + tEnd);

                    double ll = it->second.streams[l].baumWelchStep(segObs);
                    if (ll != LOG_ZERO) { totalLL += ll; ++seqCount; }
                }
            }
        }
    }
    return (seqCount > 0) ? totalLL / seqCount : LOG_ZERO;
}

// ═══════════════════════════════════════════════════════════════════════════
//  train()
// ═══════════════════════════════════════════════════════════════════════════

std::map<std::string, CharacterModel>
EmbeddedTrainer::train(
        const std::map<std::string, std::vector<ObsSequence>>& corpus,
        const std::map<std::string, std::vector<std::string>>& lexicon,
        const std::array<int, NUM_STREAMS>& dims)
{
    std::mt19937 rng(cfg_.randomSeed);

    // ── 1. Collect all unique character labels ─────────────────────────────
    std::set<std::string> charSet;
    for (auto& [w, cs] : lexicon) for (auto& c : cs) charSet.insert(c);

    // ── 2. Initialise character models ─────────────────────────────────────
    std::map<std::string, CharacterModel> models;
    for (auto& c : charSet) {
        models.emplace(c, CharacterModel(c, dims,
                                         cfg_.numStates, cfg_.numMixtures,
                                         cfg_.durType));
        // Random initialisation of emission parameters
        for (int l = 0; l < NUM_STREAMS; ++l)
            models[c].streams[l].randomInit(rng);
    }

    if (cfg_.verbose)
        std::cout << "[EmbeddedTrainer] " << models.size()
                  << " character models, " << corpus.size() << " word classes\n";

    // ── 3. EM iterations ───────────────────────────────────────────────────
    double prevLL = LOG_ZERO;
    for (int iter = 0; iter < cfg_.maxIterations; ++iter) {
        double ll = embeddedBWStep(models, corpus, lexicon);
        if (cfg_.verbose)
            std::printf("  iter %3d  avgLL = %.4f\n", iter+1, ll);

        if (prevLL != LOG_ZERO && std::abs(ll - prevLL) < cfg_.convergenceTol) {
            if (cfg_.verbose) std::cout << "  [converged]\n";
            break;
        }
        prevLL = ll;
    }

    return models;
}

// ═══════════════════════════════════════════════════════════════════════════
//  computeWeights()  (Eq. 25–26)
// ═══════════════════════════════════════════════════════════════════════════

std::map<std::string, std::array<double, NUM_STREAMS>>
EmbeddedTrainer::computeWeights(
        const std::map<std::string, CharacterModel>& models,
        const std::map<std::string, std::vector<ObsSequence>>& corpus,
        const std::map<std::string, std::vector<std::string>>& lexicon) const
{
    // Build per-character observation lists
    std::map<std::string, std::vector<ObsSequence>> charObs;
    for (auto& [wordLabel, seqs] : corpus) {
        auto lit = lexicon.find(wordLabel);
        if (lit == lexicon.end()) continue;
        const auto& charSeq = lit->second;
        for (auto& seq : seqs) {
            int T = (int)seq.size();
            int K = (int)charSeq.size();
            if (K == 0) continue;
            int seg = std::max(1, T / K);
            for (int k = 0; k < K; ++k) {
                int t0 = k*seg, t1 = (k==K-1)?T:std::min(T,(k+1)*seg);
                if (t0>=t1) continue;
                ObsSequence s(seq.begin()+t0, seq.begin()+t1);
                charObs[charSeq[k]].push_back(s);
            }
        }
    }

    std::map<std::string, std::array<double, NUM_STREAMS>> weights;
    for (auto& [charLabel, cm] : models) {
        auto it = charObs.find(charLabel);
        std::vector<ObsSequence> obs = (it != charObs.end()) ? it->second
                                                              : std::vector<ObsSequence>{};
        weights[charLabel] = StreamWeightOptimiser::compute(cm, obs);
    }
    return weights;
}

// ═══════════════════════════════════════════════════════════════════════════
//  MSDHMM
// ═══════════════════════════════════════════════════════════════════════════

void MSDHMM::train(
        const std::map<std::string, std::vector<Image>>& corpus,
        const std::map<std::string, std::vector<std::string>>& lexicon,
        const PreprocessingParams& prepParams)
{
    lexicon_ = lexicon;
    extractor_ = FeatureExtractor(prepParams);

    if (trainerCfg_.verbose)
        std::cout << "[MSDHMM] Extracting features from "
                  << corpus.size() << " word classes...\n";

    // ── Feature extraction ─────────────────────────────────────────────────
    std::map<std::string, std::vector<ObsSequence>> obsCorpus;
    for (auto& [label, images] : corpus) {
        for (auto& img : images) {
            ObsSequence seq = extractor_.extract(img);
            if (!seq.empty()) obsCorpus[label].push_back(seq);
        }
    }

    if (trainerCfg_.verbose)
        std::cout << "[MSDHMM] Starting embedded training...\n";

    // ── Train character models ─────────────────────────────────────────────
    EmbeddedTrainer trainer(trainerCfg_);
    charModels_ = trainer.train(obsCorpus, lexicon_, STREAM_DIMS);

    // ── Compute stream weights ─────────────────────────────────────────────
    if (trainerCfg_.verbose)
        std::cout << "[MSDHMM] Computing stream weights...\n";
    weights_ = trainer.computeWeights(charModels_, obsCorpus, lexicon_);

    if (trainerCfg_.verbose)
        std::cout << "[MSDHMM] Training complete. "
                  << charModels_.size() << " character models trained.\n";
}

DecodeResult MSDHMM::recognize(const Image& img,
                                const std::vector<std::string>& nBest) const
{
    ObsSequence seq = extractor_.extract(img);
    if (seq.empty()) return {"", LOG_ZERO, {}, {}};

    TwoLevelDecoder decoder(charModels_, weights_);
    return decoder.decode(seq, nBest, lexicon_);
}

// ── Simple binary serialisation ───────────────────────────────────────────

bool MSDHMM::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f) return false;
    // Write lexicon
    f << "LEXICON " << lexicon_.size() << "\n";
    for (auto& [w, cs] : lexicon_) {
        f << w << " " << cs.size();
        for (auto& c : cs) f << " " << c;
        f << "\n";
    }
    // Write character model labels only (weights)
    f << "MODELS " << charModels_.size() << "\n";
    for (auto& [label, _] : charModels_) {
        f << label << "\n";
        auto wit = weights_.find(label);
        if (wit != weights_.end()) {
            for (int l = 0; l < NUM_STREAMS; ++l)
                f << wit->second[l] << " ";
            f << "\n";
        } else {
            for (int l = 0; l < NUM_STREAMS; ++l)
                f << (1.0/NUM_STREAMS) << " ";
            f << "\n";
        }
    }
    return true;
}

bool MSDHMM::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string tok; int n;
    f >> tok >> n;
    lexicon_.clear();
    for (int i = 0; i < n; ++i) {
        std::string w; int k; f >> w >> k;
        std::vector<std::string> cs(k);
        for (auto& c : cs) f >> c;
        lexicon_[w] = cs;
    }
    f >> tok >> n;
    weights_.clear();
    for (int i = 0; i < n; ++i) {
        std::string label; f >> label;
        std::array<double, NUM_STREAMS> w;
        for (int l = 0; l < NUM_STREAMS; ++l) f >> w[l];
        weights_[label] = w;
    }
    return true;
}

} // namespace ouhawr
