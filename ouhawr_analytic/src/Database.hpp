#pragma once
/**
 * @file   Database.hpp
 * @brief  IFN/ENIT database loader and evaluation utilities.
 */

#include "Preprocessing.hpp"
#include <string>
#include <vector>
#include <map>

namespace ouhawr {

// ────────────────────────────────────────────────────────────────────────────
// Database record
// ────────────────────────────────────────────────────────────────────────────

struct WordRecord {
    std::string label;       ///< Ground-truth word transcription
    std::string imagePath;   ///< Path to binary word image
    std::string writer;      ///< Writer ID (for writer-independent split)
    std::string partition;   ///< "train" | "test"
};

// ────────────────────────────────────────────────────────────────────────────
// IFN/ENIT database loader
// ────────────────────────────────────────────────────────────────────────────

/**
 * @class IFNENITLoader
 * @brief Loads the IFN/ENIT Arabic handwriting database.
 *
 * Expected directory structure:
 *   <root>/
 *     sets/
 *       a/  b/  c/  d/  e/   – writer sets
 *         <word>.tif          – word images
 *     truth/
 *       <set>.xml             – ground-truth labels
 *
 * For evaluation on the benchmark the dataset uses sets a-d for training
 * and set e for testing (standard IFN/ENIT protocol).
 */
class IFNENITLoader {
public:
    explicit IFNENITLoader(const std::string& rootPath) : root_(rootPath) {}

    /// Load all records from the database.
    std::vector<WordRecord> loadAll() const;

    /// Load only training records (sets a–d).
    std::vector<WordRecord> loadTrain() const;

    /// Load only test records (set e).
    std::vector<WordRecord> loadTest() const;

    /// Load an image from a record (returns empty image on failure).
    Image loadImage(const WordRecord& rec) const;

    /// Build corpus map: label -> list of images (for training).
    std::map<std::string, std::vector<Image>>
    buildCorpus(const std::vector<WordRecord>& records) const;

    /// Build a simple Arabic lexicon mapping word → character allograph sequence.
    static std::map<std::string, std::vector<std::string>>
    buildLexicon(const std::vector<WordRecord>& records);

private:
    std::string root_;
    std::vector<WordRecord> loadSet(const std::string& setId) const;
};

// ────────────────────────────────────────────────────────────────────────────
// Evaluation utilities
// ────────────────────────────────────────────────────────────────────────────

struct EvalResult {
    int   total       = 0;
    int   correct     = 0;
    double accuracy   = 0.0;

    /// Top-N accuracy for N-best hypotheses.
    std::vector<std::pair<int,double>> topNAccuracy; ///< (N, acc)
};

/**
 * @class Evaluator
 * @brief Computes recognition rate on a test set.
 */
class Evaluator {
public:
    /**
     * @brief Evaluate a recognition system on a set of test records.
     *
     * @param records    Test word records (with ground-truth labels).
     * @param predicted  Predicted labels (same ordering as records).
     */
    static EvalResult evaluate(const std::vector<WordRecord>& records,
                               const std::vector<std::string>& predicted);

    /**
     * @brief Print a formatted evaluation report to stdout.
     */
    static void printReport(const EvalResult& result, const std::string& tag = "");

    /**
     * @brief Compute per-class accuracy.
     */
    static std::map<std::string, double>
    perClassAccuracy(const std::vector<WordRecord>& records,
                     const std::vector<std::string>& predicted);
};

// ────────────────────────────────────────────────────────────────────────────
// Synthetic dataset (for unit-testing without real database)
// ────────────────────────────────────────────────────────────────────────────

/**
 * @brief Generate a small synthetic dataset for testing.
 *
 * Creates N word classes, each with numSamplesPerClass images.
 * Images are noise-augmented variants of a simple prototype.
 *
 * @param numClasses          Number of distinct word classes.
 * @param numSamplesPerClass  Training samples per class.
 * @param numTestSamples      Test samples per class.
 * @param[out] corpus         Training corpus.
 * @param[out] testCorpus     Test corpus.
 * @param[out] lexicon        Word → character sequence mapping.
 */
void makeSyntheticDataset(
    int numClasses,
    int numSamplesPerClass,
    int numTestSamples,
    std::map<std::string, std::vector<Image>>& corpus,
    std::map<std::string, std::vector<Image>>& testCorpus,
    std::map<std::string, std::vector<std::string>>& lexicon,
    unsigned seed = 42);

} // namespace ouhawr
