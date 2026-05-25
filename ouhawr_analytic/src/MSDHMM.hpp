#pragma once
/**
 * @file   MSDHMM.hpp
 * @brief  Multi-Stream HMM with Explicit State Duration (MSDHMM).
 *
 * Implements the complete model described in §3 of:
 *   Benouareth et al., "Offline Handwritten Arabic Word Recognition using
 *   Multi-Stream HMM with Explicit State Duration", Information Fusion, 2026.
 *
 * Key components
 * ──────────────
 *  DurationDistribution   – abstract base; concrete: Gamma, Gaussian, Laplace, Poisson, Mixture
 *  StreamHMM              – single-stream CHMM with explicit duration (CHSMM)
 *  MSDHMM                 – synchronous multi-stream model (L=4 streams)
 *  EmbeddedTrainer        – embedded (word-level) Viterbi/Baum-Welch trainer
 *  TwoLevelDecoder        – fast two-level decoder (Algorithms 1 & 2)
 */

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cassert>
#include <random>
#include <map>
#include <functional>

#include "Preprocessing.hpp"   // ObsSequence, Frame

namespace ouhawr {

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

static constexpr double LOG_ZERO  = -1e30;
static constexpr double LOG_ONE   =  0.0;
static constexpr int    NUM_STREAMS = 4;     ///< L = 4 (paper §5)
static constexpr double MIN_PROB   = 1e-300;

inline double safeLog(double x) {
    return (x > MIN_PROB) ? std::log(x) : LOG_ZERO;
}
inline double logAdd(double a, double b) {
    if (a == LOG_ZERO) return b;
    if (b == LOG_ZERO) return a;
    if (a > b) return a + std::log1p(std::exp(b - a));
    return b + std::log1p(std::exp(a - b));
}

// ────────────────────────────────────────────────────────────────────────────
// Duration distribution interface
// ────────────────────────────────────────────────────────────────────────────

/**
 * @class DurationDistribution
 * @brief Abstract base for p_k(d) – probability of sojourning in state k for d frames.
 */
class DurationDistribution {
public:
    virtual ~DurationDistribution() = default;

    /// p(d) – probability of duration d (d >= 1).
    virtual double prob(int d) const = 0;

    /// log p(d).
    virtual double logProb(int d) const { return safeLog(prob(d)); }

    /// Maximum supported duration (for truncation in algorithms).
    virtual int maxDuration() const = 0;

    /// Re-estimate parameters from weighted occupation statistics.
    virtual void reestimate(const std::vector<double>& dWeights) = 0;

    /// Deep copy.
    virtual std::unique_ptr<DurationDistribution> clone() const = 0;

    virtual std::string name() const = 0;
};

// ────────────────────────────────────────────────────────────────────────────
// Gamma distribution  (§3.3, re-estimation from Levinson 1986 [8])
// ────────────────────────────────────────────────────────────────────────────

class GammaDuration : public DurationDistribution {
public:
    double shape; ///< α > 0
    double rate;  ///< β > 0  (mean = α/β)

    explicit GammaDuration(double shape = 2.0, double rate = 0.5)
        : shape(shape), rate(rate) {}

    double prob(int d) const override;
    int    maxDuration() const override;
    void   reestimate(const std::vector<double>& w) override;
    std::unique_ptr<DurationDistribution> clone() const override {
        return std::make_unique<GammaDuration>(*this);
    }
    std::string name() const override { return "Gamma"; }

private:
    /// log-Gamma function (Stirling approximation for large x).
    static double logGamma(double x);
    /// Digamma function (log-derivative of Gamma) via recurrence.
    static double digamma(double x);
};

// ────────────────────────────────────────────────────────────────────────────
// Gaussian distribution
// ────────────────────────────────────────────────────────────────────────────

class GaussianDuration : public DurationDistribution {
public:
    double mu;    ///< mean (location)
    double sigma; ///< standard deviation

    explicit GaussianDuration(double mu = 5.0, double sigma = 2.0)
        : mu(mu), sigma(std::max(sigma, 0.1)) {}

    double prob(int d) const override;
    int    maxDuration() const override;
    void   reestimate(const std::vector<double>& w) override;
    std::unique_ptr<DurationDistribution> clone() const override {
        return std::make_unique<GaussianDuration>(*this);
    }
    std::string name() const override { return "Gaussian"; }
};

// ────────────────────────────────────────────────────────────────────────────
// Laplace distribution  (§3.3.1, Eq. 14–18)
// ────────────────────────────────────────────────────────────────────────────

/**
 * @class LaplaceDuration
 * @brief Laplace (double-exponential) duration distribution.
 *
 * p_k(d) = (1 / 2ν) * exp(-|d - μ| / ν)
 *
 * Parameters:
 *   μ  – location (re-estimated via Eq. 18)
 *   ν  – scale / diversity (re-estimated via Eq. 17)
 *
 * Advantages (§3.3.1):
 *  • Unimodal and symmetric around the most probable duration.
 *  • Heavier tails than Gaussian – accommodates handwriting variability.
 *  • Closed-form MLE: location = sample median, scale = mean |d − μ|.
 */
class LaplaceDuration : public DurationDistribution {
public:
    double mu;  ///< location parameter (mean)
    double nu;  ///< scale parameter (diversity), nu > 0

    explicit LaplaceDuration(double mu = 5.0, double nu = 2.0)
        : mu(mu), nu(std::max(nu, 0.01)) {}

    double prob(int d) const override;
    int    maxDuration() const override;

    /**
     * @brief Re-estimate ν via Eq. (17) and μ via Eq. (18).
     *
     * @param dWeights  dWeights[d] = weighted occupation for duration d+1.
     *
     * Equation (17) – scale update:
     *   ν̄_k  =  Σ_t Σ_d |d − μ_k| · ξ_t(k,d)  /  Σ_t α_t(k) β_t(k)
     * Since the caller passes collapsed duration weights, we approximate:
     *   ν̄  =  Σ_d |d − μ| · w_d  /  Σ_d w_d
     *
     * Equation (18) – location update (maximiser of weighted occupancy):
     *   μ̄_k  =  arg max_d  [ Σ_m α_{T-d}(m) a_{mk} p_k(d) Π b_k(o_s) ]
     * Approximated as the weighted median of d values.
     */
    void reestimate(const std::vector<double>& w) override;

    std::unique_ptr<DurationDistribution> clone() const override {
        return std::make_unique<LaplaceDuration>(*this);
    }
    std::string name() const override { return "Laplace"; }
};

// ────────────────────────────────────────────────────────────────────────────
// Poisson distribution
// ────────────────────────────────────────────────────────────────────────────

class PoissonDuration : public DurationDistribution {
public:
    double lambda; ///< rate parameter

    explicit PoissonDuration(double lambda = 4.0)
        : lambda(std::max(lambda, 0.1)) {}

    double prob(int d) const override;
    int    maxDuration() const override;
    void   reestimate(const std::vector<double>& w) override;
    std::unique_ptr<DurationDistribution> clone() const override {
        return std::make_unique<PoissonDuration>(*this);
    }
    std::string name() const override { return "Poisson"; }

private:
    /// Cached log-factorial.
    static double logFactorial(int n);
};

// ────────────────────────────────────────────────────────────────────────────
// Mixture of Gamma and Laplace  (§3.3.2, Eq. 19–22)
// ────────────────────────────────────────────────────────────────────────────

/**
 * @class MixtureDuration
 * @brief p_k(d) = c_{k1} * p^1_k(d) + c_{k2} * p^2_k(d)
 *
 * Default: component 0 = Gamma, component 1 = Laplace (paper §3.3.2).
 * Generalises to arbitrary M via Eq. (21)–(22).
 *
 * Mixture coefficient re-estimation (Eq. 21):
 *   c̄_{km} = Σ_t Σ_d ξ_t(k,d,m) / Σ_t Σ_d Σ_m ξ_t(k,d,m)
 *
 * where ξ_t(k,d,m) ∝ c_{km} * p^m_k(d) (see Eq. 22).
 */
class MixtureDuration : public DurationDistribution {
public:
    std::vector<std::unique_ptr<DurationDistribution>> components;
    std::vector<double> coefficients; ///< c_{k1}, c_{k2}, … (sum to 1)

    MixtureDuration();
    MixtureDuration(const MixtureDuration& o);
    MixtureDuration& operator=(const MixtureDuration& o);

    double prob(int d) const override;
    int    maxDuration() const override;

    /**
     * @brief Re-estimate mixture coefficients (Eq. 21–22) and component params.
     *
     * @param dWeights  dWeights[d] = total weighted occupation for duration d+1.
     *
     * The responsibility of component m for duration d is:
     *   r_m(d) = c_m * p^m(d) / Σ_m c_m * p^m(d)
     *
     * Updated coefficients (Eq. 21):
     *   c̄_m = Σ_d r_m(d) * w_d / Σ_d w_d
     *
     * Component-specific weights passed to each sub-distribution:
     *   w^m_d = r_m(d) * w_d
     */
    void reestimate(const std::vector<double>& w) override;

    std::unique_ptr<DurationDistribution> clone() const override;
    std::string name() const override { return "Mixture(Gamma+Laplace)"; }
};

// ────────────────────────────────────────────────────────────────────────────
// Gaussian emission (single-stream)
// ────────────────────────────────────────────────────────────────────────────

/// Single Gaussian component.
struct GaussComp {
    std::vector<double> mean;
    std::vector<double> var;   ///< diagonal covariance
    double weight = 1.0;

    GaussComp() = default;
    GaussComp(int dim, double initVar = 1.0);

    double logLikelihood(const std::vector<double>& obs) const;
};

/// Mixture of Gaussians emission PDF for one HMM state.
struct GMMEmission {
    std::vector<GaussComp> components;
    int dim = 0;

    GMMEmission() = default;
    explicit GMMEmission(int dim, int M = 3);

    double logLikelihood(const std::vector<double>& obs) const;

    /// Baum-Welch accumulate.
    void accumulate(const std::vector<double>& obs, double gamma,
                    std::vector<GaussComp>& accMean,
                    std::vector<double>&    accWeight,
                    std::vector<GaussComp>& accVar) const;

    /// Update from accumulators.
    void update(const std::vector<GaussComp>& accMean,
                const std::vector<double>&    accWeight,
                const std::vector<GaussComp>& accVar);
};

// ────────────────────────────────────────────────────────────────────────────
// StreamHMM – single-stream CHSMM for one character shape
// ────────────────────────────────────────────────────────────────────────────

/**
 * @class StreamHMM
 * @brief Continuous HMM with explicit state duration for one observation stream.
 *
 * Topology (§5.3): right-to-left, 4 states, self-transitions omitted,
 * optional skip of one state.
 *
 * Duration distributions: one per state, selectable type.
 */
class StreamHMM {
public:
    // ── Model parameters ──────────────────────────────────────────────────
    int N;             ///< Number of states (default 4)
    int M;             ///< Gaussian mixture components per state (default 3)
    int dim;           ///< Feature dimension for this stream

    std::vector<double>              pi;  ///< Initial state distribution [N]
    std::vector<std::vector<double>> A;   ///< Transition matrix [N][N]
    std::vector<GMMEmission>         B;   ///< Emission PDFs [N]
    std::vector<std::unique_ptr<DurationDistribution>> dur; ///< Duration PDFs [N]

    StreamHMM() = default;
    StreamHMM(int states, int gaussMix, int featureDim,
              const std::string& durType = "Gamma");
    StreamHMM(const StreamHMM& o);
    StreamHMM& operator=(const StreamHMM& o);

    // ── Inference ──────────────────────────────────────────────────────────

    /**
     * @brief Forward-backward algorithm with explicit duration (Eq. 8–12).
     * @param obs  Observation sequence (T frames, each dim-dimensional).
     * @param[out] alpha  Forward variables [T+1][N].
     * @param[out] beta   Backward variables [T+1][N].
     * @return log P(O | λ)
     */
    double forwardBackward(const std::vector<std::vector<double>>& obs,
                           std::vector<std::vector<double>>& alpha,
                           std::vector<std::vector<double>>& beta) const;

    /**
     * @brief Viterbi decoding with explicit duration.
     * @param obs      Observation sequence.
     * @param[out] bestPath  Optimal state sequence [T].
     * @param[out] bestDurs  Duration of each state visit.
     * @return log P*(O | λ)
     */
    double viterbi(const std::vector<std::vector<double>>& obs,
                   std::vector<int>& bestPath,
                   std::vector<int>& bestDurs) const;

    /**
     * @brief Level-1 decoding kernel used by TwoLevelDecoder (Algorithm 1).
     *
     * Computes best-score matrix χ[s][e] and best-path matrix ε[s][e]
     * for all (start, end) pairs, incorporating explicit duration.
     *
     * @param obs    Full stream observation sequence.
     * @param chi    [T][T] – best log-probability for segment [s,e].
     * @param epsilon [T][T] – best state sequence for segment [s,e].
     */
    void level1Decode(const std::vector<std::vector<double>>& obs,
                      std::vector<std::vector<double>>&              chi,
                      std::vector<std::vector<std::vector<int>>>&   epsilon) const;

    // ── Training ───────────────────────────────────────────────────────────

    /**
     * @brief Single Baum-Welch iteration on one observation sequence.
     * @return log-likelihood before update.
     */
    double baumWelchStep(const std::vector<std::vector<double>>& obs);

    /**
     * @brief Single Viterbi training iteration (hard assignment).
     */
    double viterbiTrainStep(const std::vector<std::vector<double>>& obs);

    // ── Utilities ──────────────────────────────────────────────────────────

    void   randomInit(std::mt19937& rng);
    void   normalizeA();
    double emissionLogProb(int state, const std::vector<double>& o) const;

    int    maxDur() const;

private:
    void   initDurations(const std::string& durType);
};

// ────────────────────────────────────────────────────────────────────────────
// CharacterModel – L-stream model for one Arabic character allograph
// ────────────────────────────────────────────────────────────────────────────

struct CharacterModel {
    std::string label;                     ///< e.g. "alef_initial"
    std::array<StreamHMM, NUM_STREAMS> streams;

    CharacterModel() = default;
    CharacterModel(const std::string& lbl,
                   const std::array<int, NUM_STREAMS>& dims,
                   int states = 4, int mix = 3,
                   const std::string& durType = "Mixture");
};

// ────────────────────────────────────────────────────────────────────────────
// Stream-weight optimisation  (§3.5, Eq. 25–26)
// ────────────────────────────────────────────────────────────────────────────

/**
 * @class StreamWeightOptimiser
 * @brief Computes per-character, per-stream relevance weights via mutual information.
 *
 * w_{lc} = I(O^l_c, Q^l_c) / Σ_l I(O^l_c, Q^l_c)     (Eq. 25)
 *
 * where I(O,Q) = H(Q) − H(Q|O)                          (Eq. 26)
 */
class StreamWeightOptimiser {
public:
    /**
     * @brief Compute stream weights for a character given its training data.
     *
     * @param charModel  Trained character model.
     * @param trainObs   Training observation sequences for this character.
     * @return           [NUM_STREAMS] weight vector (sums to 1).
     */
    static std::array<double, NUM_STREAMS>
    compute(const CharacterModel& charModel,
            const std::vector<ObsSequence>& trainObs);

private:
    static double mutualInfo(const StreamHMM& hmm,
                             const std::vector<std::vector<double>>& obs);
    static double stateEntropy(const std::vector<double>& occ);
};

// ────────────────────────────────────────────────────────────────────────────
// Two-level decoder  (§3.4, Algorithms 1 & 2)
// ────────────────────────────────────────────────────────────────────────────

/// Result of decoding one word.
struct DecodeResult {
    std::string word;              ///< Best matching word
    double      logLikelihood;     ///< Log P(S | v*)
    std::vector<std::string> charSeq;  ///< Character sequence
    /// Per-stream, per-character best state path.
    std::vector<std::array<std::vector<int>, NUM_STREAMS>> statePaths;
};

/**
 * @class TwoLevelDecoder
 * @brief Fast two-level MSHMM decoding with explicit state duration.
 *
 * Level 1 (Algorithm 1):
 *   For each character model c and stream l, compute:
 *     χ^{cl}(s,e) – best log-probability over segment [s,e]
 *     ε^{cl}(s,e) – best state sequence over segment [s,e]
 *   Complexity: O(T³ N² L C)
 *
 * Level 2 (Algorithm 2):
 *   For each word v in the N-best hypothesis list L:
 *     δ̂_t(k) = best log-probability ending at char k at time t
 *   Complexity: O(LC + T² K V)
 *   Overall: O(T³ N² C L + T² K V)
 */
class TwoLevelDecoder {
public:
    /**
     * @param charModels  All trained character models.
     * @param weights     weights[charLabel][stream] – relevance weights.
     */
    explicit TwoLevelDecoder(
        const std::map<std::string, CharacterModel>& charModels,
        const std::map<std::string, std::array<double, NUM_STREAMS>>& weights);

    /**
     * @brief Decode a word image from N-best hypotheses (Algorithm 1 + 2).
     *
     * @param obs      Multi-stream observation sequence for the word image.
     * @param nBest    N-best candidate word strings (from holistic stage).
     * @param lexicon  Word-to-character-sequence mapping.
     * @return         Best matching word with its score and segmentation.
     */
    DecodeResult decode(const ObsSequence& obs,
                        const std::vector<std::string>& nBest,
                        const std::map<std::string, std::vector<std::string>>& lexicon) const;

private:
    // ── Level 1 precomputation ─────────────────────────────────────────────

    struct Level1Cache {
        /// chi[charLabel][l][s][e] = log best-score for stream l, segment [s,e]
        std::map<std::string,
            std::array<std::vector<std::vector<double>>, NUM_STREAMS>> chi;
        /// epsilon[charLabel][l][s][e] = best state sequence
        std::map<std::string,
            std::array<std::vector<std::vector<std::vector<int>>>, NUM_STREAMS>> epsilon;
        int T = 0;
    };

    /**
     * @brief Algorithm 1: pre-compute χ and ε for all characters and streams.
     */
    Level1Cache runLevel1(const ObsSequence& obs,
                          const std::vector<std::string>& candidates,
                          const std::map<std::string, std::vector<std::string>>& lexicon) const;

    /**
     * @brief Combine per-stream χ scores using stream weights.
     *        χ^c(s,e) = Σ_l w^l_c * χ^{cl}(s,e)
     */
    std::vector<std::vector<double>>
    combineStreams(const std::string& charLabel,
                  const std::array<std::vector<std::vector<double>>, NUM_STREAMS>& chi,
                  int T) const;

    /**
     * @brief Algorithm 2: decode one word using pre-computed combined scores.
     */
    std::pair<double, std::vector<int>>
    runLevel2Word(const std::vector<std::string>& charSeq,
                  const std::map<std::string, std::vector<std::vector<double>>>& combinedChi,
                  int T) const;

    const std::map<std::string, CharacterModel>&                        models_;
    const std::map<std::string, std::array<double, NUM_STREAMS>>&       weights_;
};

// ────────────────────────────────────────────────────────────────────────────
// Embedded trainer  (§5.3)
// ────────────────────────────────────────────────────────────────────────────

struct TrainingConfig {
    int    maxIterations    = 30;     ///< EM iterations
    double convergenceTol   = 1e-4;   ///< Log-likelihood change threshold
    int    numStates        = 4;      ///< States per character HMM
    int    numMixtures      = 3;      ///< Gaussian mixture components
    std::string durType     = "Mixture"; ///< Duration distribution type
    bool   verbose          = true;
    unsigned randomSeed     = 42;
};

/**
 * @class EmbeddedTrainer
 * @brief Trains character-level CHSMMs from word-level samples.
 *
 * Embedded training (§5.3):
 *   – Character models are concatenated to form word models.
 *   – The Viterbi/Baum-Welch algorithm is run on whole-word observations.
 *   – No explicit segmentation is needed.
 *   – Handles Arabic positional allography: up to 4 shapes × 28 characters = 112+ models.
 */
class EmbeddedTrainer {
public:
    explicit EmbeddedTrainer(const TrainingConfig& cfg = {}) : cfg_(cfg) {}

    /**
     * @brief Train all character models from a labelled corpus.
     *
     * @param corpus   corpus[wordLabel] = list of ObsSequences for that word.
     * @param lexicon  Maps word label → ordered list of character allograph labels.
     * @param dims     Feature dimension for each stream.
     * @return         Map from character allograph label → trained CharacterModel.
     */
    std::map<std::string, CharacterModel>
    train(const std::map<std::string, std::vector<ObsSequence>>& corpus,
          const std::map<std::string, std::vector<std::string>>& lexicon,
          const std::array<int, NUM_STREAMS>& dims);

    /**
     * @brief Compute stream weights for all characters after training.
     */
    std::map<std::string, std::array<double, NUM_STREAMS>>
    computeWeights(
        const std::map<std::string, CharacterModel>& models,
        const std::map<std::string, std::vector<ObsSequence>>& corpus,
        const std::map<std::string, std::vector<std::string>>& lexicon) const;

private:
    TrainingConfig cfg_;

    /// One Baum-Welch iteration over the whole corpus (all words, all streams).
    double embeddedBWStep(
        std::map<std::string, CharacterModel>& models,
        const std::map<std::string, std::vector<ObsSequence>>& corpus,
        const std::map<std::string, std::vector<std::string>>& lexicon) const;

    /// Extract the stream-l observations for a full word ObsSequence.
    static std::vector<std::vector<double>>
    getStreamObs(const ObsSequence& seq, int streamIdx);
};

// ────────────────────────────────────────────────────────────────────────────
// MSDHMM – top-level recognition system  (§5)
// ────────────────────────────────────────────────────────────────────────────

/**
 * @class MSDHMM
 * @brief Full analytic OUHAWR system (§5).
 *
 * Usage:
 *   1. train()  – embedded training from labelled word images.
 *   2. decode() – recognise a word image given N-best hypotheses.
 */
class MSDHMM {
public:
    explicit MSDHMM(const TrainingConfig& cfg = {}) : trainerCfg_(cfg) {}

    // ── Training ───────────────────────────────────────────────────────────

    /**
     * @brief Train the system from a labelled corpus of word images.
     *
     * @param corpus   corpus[wordLabel] = list of word images.
     * @param lexicon  Maps word label → ordered list of character allograph labels.
     */
    void train(const std::map<std::string, std::vector<Image>>& corpus,
               const std::map<std::string, std::vector<std::string>>& lexicon,
               const PreprocessingParams& prepParams = {});

    // ── Recognition ────────────────────────────────────────────────────────

    /**
     * @brief Recognise a word image.
     *
     * @param img       Input word image.
     * @param nBest     N-best hypotheses from the holistic pre-filtering stage.
     * @return          Best matching word and its log-likelihood.
     */
    DecodeResult recognize(const Image& img,
                           const std::vector<std::string>& nBest) const;

    // ── Persistence ────────────────────────────────────────────────────────

    bool save(const std::string& path) const;
    bool load(const std::string& path);

    // ── Accessors ──────────────────────────────────────────────────────────

    const std::map<std::string, CharacterModel>& characterModels() const { return charModels_; }
    const std::map<std::string, std::array<double, NUM_STREAMS>>& streamWeights() const { return weights_; }

private:
    TrainingConfig  trainerCfg_;
    FeatureExtractor extractor_;

    std::map<std::string, CharacterModel>               charModels_;
    std::map<std::string, std::array<double, NUM_STREAMS>> weights_;
    std::map<std::string, std::vector<std::string>>     lexicon_;

    /// Feature dimensions for the four streams.
    static constexpr std::array<int, NUM_STREAMS> STREAM_DIMS = {15, 15, 26, 24};
};

} // namespace ouhawr
