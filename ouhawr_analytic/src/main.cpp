/**
 * @file   main.cpp
 * @brief  Test driver for the OUHAWR MSDHMM analytic recognition system.
 *
 * Demonstrates:
 *   1. Synthetic dataset training & evaluation (no database needed)
 *   2. IFN/ENIT database loading & evaluation (if path provided via --db flag)
 *
 * Usage:
 *   ./ouhawr_test [--db <ifnenit_root>] [--dur <Gamma|Laplace|Mixture|Poisson>]
 *                 [--iter <N>] [--nbest <N>] [--classes <N>] [--verbose]
 */

#include "MSDHMM.hpp"
#include "Preprocessing.hpp"
#include "Database.hpp"

#include <iostream>
#include <string>
#include <chrono>
#include <map>
#include <vector>
#include <algorithm>
#include <random>

using namespace ouhawr;

// ─── Helpers ────────────────────────────────────────────────────────────────

static void printBanner() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════╗
║  OUHAWR – Offline Handwritten Arabic Word Recognition            ║
║  Multi-Stream HMM with Explicit State Duration                   ║
║  Benouareth et al., Information Fusion 2026                      ║
╚══════════════════════════════════════════════════════════════════╝
)" << "\n";
}

static void printConfig(const TrainingConfig& cfg, const std::string& durType,
                         int nBest, int numClasses) {
    std::cout << "━━━ Configuration ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Duration distribution : " << durType << "\n";
    std::cout << "  HMM states per char   : " << cfg.numStates   << "\n";
    std::cout << "  Gaussian mixtures     : " << cfg.numMixtures << "\n";
    std::cout << "  EM iterations (max)   : " << cfg.maxIterations << "\n";
    std::cout << "  N-best hypotheses     : " << nBest << "\n";
    std::cout << "  Word classes          : " << numClasses << "\n";
    std::cout << "  Streams (L)           : " << NUM_STREAMS << "\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
}

// ─── Duration distribution comparison ───────────────────────────────────────

static void runDurationComparisonTest() {
    std::cout << "\n━━━ Duration Distribution Unit Test ━━━━━━━━━━━━━━━━━━━━━━\n";

    // Test each distribution
    auto test = [](const std::string& name, DurationDistribution& dist) {
        double sum = 0;
        for (int d = 1; d <= dist.maxDuration(); ++d) sum += dist.prob(d);
        std::printf("  %-25s  max_d=%3d  sum(p)=%.4f\n",
                    name.c_str(), dist.maxDuration(), sum);

        // Re-estimation test with synthetic weights
        std::vector<double> w(dist.maxDuration(), 0.0);
        for (int d = 1; d <= (int)w.size(); ++d) w[d-1] = dist.prob(d);
        dist.reestimate(w);
    };

    GammaDuration    gamma(2.0, 0.5);   test("Gamma(shape=2, rate=0.5)", gamma);
    GaussianDuration gauss(5.0, 2.0);   test("Gaussian(mu=5, sigma=2)", gauss);
    LaplaceDuration  lap(5.0, 1.5);     test("Laplace(mu=5, nu=1.5)", lap);
    PoissonDuration  pois(4.0);         test("Poisson(lambda=4)", pois);
    MixtureDuration  mix;               test("Mixture(Gamma+Laplace)", mix);

    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
}

// ─── Feature extraction unit test ───────────────────────────────────────────

static void runFeatureExtractionTest() {
    std::cout << "\n━━━ Feature Extraction Unit Test ━━━━━━━━━━━━━━━━━━━━━━━━\n";

    Image img = makeSyntheticWordImage(120, 40, 42);
    FeatureExtractor extractor;
    ObsSequence seq = extractor.extract(img);

    std::printf("  Image size     : %d × %d\n", img.width, img.height);
    std::printf("  Frames         : %d\n", (int)seq.size());
    if (!seq.empty()) {
        std::printf("  Stream dims    : upper=%d  lower=%d  stat=%d  struct=%d\n",
                    (int)seq[0].upperContour.size(),
                    (int)seq[0].lowerContour.size(),
                    (int)seq[0].statistical.size(),
                    (int)seq[0].structural.size());
    }
    std::cout << "  [OK]\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
}

// ─── Synthetic dataset run ───────────────────────────────────────────────────

static void runSyntheticTest(const TrainingConfig& cfg, int nBest, int numClasses,
                              int numTrain, int numTest) {
    std::cout << "\n━━━ Synthetic Dataset Experiment ━━━━━━━━━━━━━━━━━━━━━━━━\n";

    // Build synthetic data
    std::map<std::string, std::vector<Image>> trainCorpus, testCorpus;
    std::map<std::string, std::vector<std::string>> lexicon;
    makeSyntheticDataset(numClasses, numTrain, numTest,
                          trainCorpus, testCorpus, lexicon, 42);

    std::cout << "  Classes: " << numClasses
              << "  |  Train/class: " << numTrain
              << "  |  Test/class: " << numTest << "\n\n";

    // Train
    auto t0 = std::chrono::steady_clock::now();
    MSDHMM system(cfg);
    system.train(trainCorpus, lexicon);
    auto t1 = std::chrono::steady_clock::now();
    double trainSec = std::chrono::duration<double>(t1 - t0).count();
    std::printf("  Training time  : %.2f s\n", trainSec);

    // Build word list
    std::vector<std::string> allWords;
    for (auto& [w, _] : lexicon) allWords.push_back(w);

    // Evaluate
    std::mt19937 rng(99);
    std::vector<WordRecord> testRecords;
    std::vector<Image>      testImages;
    std::vector<std::string> gtLabels;

    for (auto& [label, images] : testCorpus) {
        for (auto& img : images) {
            WordRecord r;
            r.label = label; r.partition = "test";
            testRecords.push_back(r);
            testImages.push_back(img);
            gtLabels.push_back(label);
        }
    }

    auto t2 = std::chrono::steady_clock::now();
    std::vector<std::string> predicted;
    int testCount = 0;

    for (int i = 0; i < (int)testImages.size(); ++i) {
        // Build N-best: correct + random distractors
        std::vector<std::string> nbest;
        nbest.push_back(gtLabels[i]);
        std::vector<std::string> others = allWords;
        others.erase(std::remove(others.begin(), others.end(), gtLabels[i]), others.end());
        std::shuffle(others.begin(), others.end(), rng);
        for (int j = 0; j < std::min(nBest-1, (int)others.size()); ++j)
            nbest.push_back(others[j]);
        std::shuffle(nbest.begin(), nbest.end(), rng);

        DecodeResult res = system.recognize(testImages[i], nbest);
        predicted.push_back(res.word.empty() ? nbest[0] : res.word);
        ++testCount;
        if (testCount % 20 == 0)
            std::printf("  [%3d / %3d] ...\r", testCount, (int)testImages.size());
    }

    auto t3 = std::chrono::steady_clock::now();
    double recogSec = std::chrono::duration<double>(t3 - t2).count();

    std::cout << "\n";
    EvalResult res = Evaluator::evaluate(testRecords, predicted);
    Evaluator::printReport(res, "Synthetic Dataset");
    std::printf("  Recognition time: %.2f s  (%.1f ms/word)\n\n",
                recogSec, 1000.0 * recogSec / std::max(1, testCount));
}

// ─── IFN/ENIT database run ───────────────────────────────────────────────────

static void runIFNENITTest(const std::string& dbPath, const TrainingConfig& cfg, int nBest) {
    std::cout << "\n━━━ IFN/ENIT Benchmark Experiment ━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Database root: " << dbPath << "\n";

    IFNENITLoader loader(dbPath);

    std::cout << "  Loading training records...\n";
    auto trainRecords = loader.loadTrain();
    std::cout << "  Training records: " << trainRecords.size() << "\n";

    if (trainRecords.empty()) {
        std::cout << "  [WARNING] No training records found. "
                     "Check database path and file format.\n";
        return;
    }

    std::cout << "  Loading test records...\n";
    auto testRecords = loader.loadTest();
    std::cout << "  Test records: " << testRecords.size() << "\n";

    // Build corpus and lexicon
    std::cout << "  Building corpus...\n";
    auto corpus  = loader.buildCorpus(trainRecords);
    auto lexicon = IFNENITLoader::buildLexicon(trainRecords);
    std::cout << "  Word classes: " << corpus.size() << "\n";
    std::cout << "  Lexicon size: " << lexicon.size() << "\n";

    // Train
    std::cout << "\n  Training MSDHMM...\n";
    auto t0 = std::chrono::steady_clock::now();
    MSDHMM system(cfg);
    system.train(corpus, lexicon);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("  Training time: %.2f s\n",
                std::chrono::duration<double>(t1 - t0).count());

    // Save model
    system.save(dbPath + "/msdhmm_model.txt");
    std::cout << "  Model saved to: " << dbPath + "/msdhmm_model.txt\n";

    // Evaluate on test set
    std::cout << "\n  Evaluating on test set...\n";
    std::vector<std::string> allWords;
    for (auto& [w, _] : lexicon) allWords.push_back(w);
    std::mt19937 rng(123);

    std::vector<std::string> predicted;
    auto t2 = std::chrono::steady_clock::now();
    int processed = 0;

    for (auto& rec : testRecords) {
        Image img = loader.loadImage(rec);
        if (img.width == 0) { predicted.push_back(""); ++processed; continue; }

        // Simulate N-best (correct + random distractors from lexicon)
        std::vector<std::string> nbest;
        nbest.push_back(rec.label);
        std::vector<std::string> others = allWords;
        others.erase(std::remove(others.begin(), others.end(), rec.label), others.end());
        std::shuffle(others.begin(), others.end(), rng);
        for (int j = 0; j < std::min(nBest-1, (int)others.size()); ++j)
            nbest.push_back(others[j]);
        std::shuffle(nbest.begin(), nbest.end(), rng);

        DecodeResult res = system.recognize(img, nbest);
        predicted.push_back(res.word.empty() ? nbest[0] : res.word);
        ++processed;

        if (processed % 50 == 0)
            std::printf("  [%4d / %4d]\r", processed, (int)testRecords.size());
    }

    auto t3 = std::chrono::steady_clock::now();
    double recogSec = std::chrono::duration<double>(t3 - t2).count();
    std::cout << "\n";

    EvalResult er = Evaluator::evaluate(testRecords, predicted);
    Evaluator::printReport(er, "IFN/ENIT Benchmark");
    std::printf("  Recognition time: %.2f s  (%.1f ms/word)\n\n",
                recogSec, 1000.0 * recogSec / std::max(1, processed));

    // Per-class accuracy (top 10 best + worst)
    auto perClass = Evaluator::perClassAccuracy(testRecords, predicted);
    std::vector<std::pair<std::string,double>> sorted(perClass.begin(), perClass.end());
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b){ return a.second > b.second; });

    std::cout << "  Top-5 best recognized classes:\n";
    for (int i = 0; i < std::min(5, (int)sorted.size()); ++i)
        std::printf("    %-25s  %.1f%%\n", sorted[i].first.c_str(), sorted[i].second);

    std::cout << "  Top-5 worst recognized classes:\n";
    for (int i = (int)sorted.size()-1; i >= std::max(0,(int)sorted.size()-5); --i)
        std::printf("    %-25s  %.1f%%\n", sorted[i].first.c_str(), sorted[i].second);
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    printBanner();

    // Default configuration
    TrainingConfig cfg;
    cfg.maxIterations = 20;
    cfg.numStates     = 4;
    cfg.numMixtures   = 3;
    cfg.durType       = "Mixture";
    cfg.verbose       = false;
    cfg.convergenceTol = 1e-3;

    std::string dbPath;
    int nBest      = 5;
    int numClasses = 8;
    int numTrain   = 10;
    int numTest    = 5;
    bool unitTests = true;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if      (arg == "--db"      && i+1 < argc) { dbPath = argv[++i]; }
        else if (arg == "--dur"     && i+1 < argc) { cfg.durType = argv[++i]; }
        else if (arg == "--iter"    && i+1 < argc) { cfg.maxIterations = std::stoi(argv[++i]); }
        else if (arg == "--nbest"   && i+1 < argc) { nBest = std::stoi(argv[++i]); }
        else if (arg == "--classes" && i+1 < argc) { numClasses = std::stoi(argv[++i]); }
        else if (arg == "--train"   && i+1 < argc) { numTrain = std::stoi(argv[++i]); }
        else if (arg == "--test"    && i+1 < argc) { numTest = std::stoi(argv[++i]); }
        else if (arg == "--verbose")                { cfg.verbose = true; }
        else if (arg == "--no-unit")                { unitTests = false; }
        else if (arg == "--help") {
            std::cout << "Usage: ouhawr_test [options]\n"
                      << "  --db <path>       IFN/ENIT root directory\n"
                      << "  --dur <type>      Duration: Gamma|Gaussian|Laplace|Poisson|Mixture\n"
                      << "  --iter <N>        Max EM iterations (default: 20)\n"
                      << "  --nbest <N>       N-best hypotheses (default: 5)\n"
                      << "  --classes <N>     Synthetic classes (default: 8)\n"
                      << "  --train <N>       Train samples/class (default: 10)\n"
                      << "  --test <N>        Test samples/class (default: 5)\n"
                      << "  --verbose         Enable verbose training output\n"
                      << "  --no-unit         Skip unit tests\n";
            return 0;
        }
    }

    printConfig(cfg, cfg.durType, nBest, numClasses);

    // ── Unit tests ─────────────────────────────────────────────────────────
    if (unitTests) {
        runDurationComparisonTest();
        runFeatureExtractionTest();
    }

    // ── Synthetic test ──────────────────────────────────────────────────────
    runSyntheticTest(cfg, nBest, numClasses, numTrain, numTest);

    // ── IFN/ENIT test ───────────────────────────────────────────────────────
    if (!dbPath.empty()) {
        runIFNENITTest(dbPath, cfg, nBest);
    } else {
        std::cout << "  [INFO] No database path given (--db). "
                     "Skipping IFN/ENIT benchmark.\n";
        std::cout << "  [INFO] To run on IFN/ENIT: ./ouhawr_test --db /path/to/ifnenit\n\n";
    }

    return 0;
}
