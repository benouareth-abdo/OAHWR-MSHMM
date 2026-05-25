#pragma once
/**
 * @file   Preprocessing.hpp
 * @brief  Pre-processing and feature-extraction for the analytic OUHAWR approach.
 *
 * Implements the four-stage pipeline described in §5 of:
 *   Benouareth et al., "Offline Handwritten Arabic Word Recognition using
 *   Multi-Stream HMM with Explicit State Duration", Information Fusion, 2026.
 *
 * Four feature streams are produced for each sliding-window frame:
 *   Stream 0 – upper-contour features  (15-dim)
 *   Stream 1 – lower-contour features  (15-dim)
 *   Stream 2 – statistical/density features (26-dim)
 *   Stream 3 – structural skeleton features (21-dim)
 */

#include <vector>
#include <array>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

namespace ouhawr {

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

/// Grey-level image stored row-major, values in [0,255].
struct Image {
    int width  = 0;
    int height = 0;
    std::vector<uint8_t> data; ///< row-major, data[y*width+x]

    Image() = default;
    Image(int w, int h) : width(w), height(h), data(w * h, 0) {}

    uint8_t  at(int x, int y) const { return data[y * width + x]; }
    uint8_t& at(int x, int y)       { return data[y * width + x]; }

    bool isForeground(int x, int y) const { return at(x, y) ==1; }
    bool valid(int x, int y) const { return x >= 0 && x < width && y >= 0 && y < height; }
};

/// A single multi-stream observation frame produced by the sliding window.
struct Frame {
    std::vector<double> upperContour;   ///< Stream 0 – 15-dim
    std::vector<double> lowerContour;   ///< Stream 1 – 15-dim
    std::vector<double> statistical;    ///< Stream 2 – 26-dim
    std::vector<double> structural;     ///< Stream 3 – 24-dim

    /// Returns the feature vector for stream l (0..3).
    const std::vector<double>& stream(int l) const {
        switch (l) {
            case 0: return upperContour;
            case 1: return lowerContour;
            case 2: return statistical;
            case 3: return structural;
            default: throw std::out_of_range("stream index out of range");
        }
    }
};

/// Observation sequence: ordered list of frames (right-to-left scan).
using ObsSequence = std::vector<Frame>;

// ────────────────────────────────────────────────────────────────────────────
// Preprocessing parameters
// ────────────────────────────────────────────────────────────────────────────

struct PreprocessingParams {
    bool   uniformSegmentation = true; ///< Fixed vs. variable frame width
    int    frameWidth          = 6;    ///< Pixels per frame (uniform mode)
    int    frameStep           = 3;    ///< Stride between frames
    int    smoothRadius        = 1;    ///< Box filter radius
    int    thinningIter        = 5;    ///< Pavlidis algorithm iterations
    double upperZoneFraction   = 0.30; ///< Fraction of height for upper zone
    double lowerZoneFraction   = 0.30; ///< Fraction of height for lower zone
};

// ────────────────────────────────────────────────────────────────────────────
// Preprocessor class
// ────────────────────────────────────────────────────────────────────────────

/**
 * @class Preprocessor
 * @brief Applies the four preprocessing stages of §5.1.
 */
class Preprocessor {
public:
    explicit Preprocessor(const PreprocessingParams& p = {}) : params_(p) {}

    Image process(const Image& src) const;
    Image smooth(const Image& src) const;
    Image thin(const Image& src) const;
    void  estimateBaselines(const Image& img, int& upper, int& lower) const;
    std::vector<int> chainCode(const Image& img) const;

    const PreprocessingParams& params() const { return params_; }

private:
    PreprocessingParams params_;

    static constexpr int DX8[8] = {1, 1, 0,-1,-1,-1, 0, 1};
    static constexpr int DY8[8] = {0,-1,-1,-1, 0, 1, 1, 1};

    bool isContourPixel(const Image& img, int x, int y) const;
    bool canRemove(const Image& img, int x, int y) const;
};

// ────────────────────────────────────────────────────────────────────────────
// Feature extraction parameters
// ────────────────────────────────────────────────────────────────────────────

struct FeatureParams {
    int contourDirBins    = 8;
    int contourZones      = 3;
    int statCellRows      = 4;
    int concavityFeatures = 10;
    int skelZones         = 3;
    int skelFeatureTypes  = 7;
};

// ────────────────────────────────────────────────────────────────────────────
// FeatureExtractor class
// ────────────────────────────────────────────────────────────────────────────

/**
 * @class FeatureExtractor
 * @brief Converts a word image into a sequence of multi-stream frames (§5.2).
 *
 * Sliding window scans right-to-left. At each position four feature vectors
 * are computed (one per stream).
 */
class FeatureExtractor {
public:
    explicit FeatureExtractor(const PreprocessingParams& pp  = {},
                              const FeatureParams&       fp  = {})
        : prep_(pp), fp_(fp) {}

    ObsSequence extract(const Image& raw) const;

    /// Stream 0 & 1: 15-dim contour features (direction histogram + zone encoding).
    std::vector<double> extractContourFeatures(const Image& proc,
                                               const std::vector<int>& cc,
                                               int xStart, int xEnd,
                                               int upperBase, int lowerBase,
                                               bool upper) const;

    /// Stream 2: 26 statistical/density features.
    std::vector<double> extractStatistical(const Image& proc,
                                           int xStart, int xEnd,
                                           int upperBase, int lowerBase) const;

    /// Stream 3: 21 structural skeleton features.
    std::vector<double> extractStructural(const Image& thinned,
                                          int xStart, int xEnd,
                                          int upperBase, int lowerBase) const;

    const PreprocessingParams& prepParams() const { return prep_.params(); }

private:
    Preprocessor  prep_;
    FeatureParams fp_;

    std::vector<double> contourDirectionHistogram(const std::vector<int>& cc,
                                                  int xStart, int xEnd) const;

    std::vector<double> contourZoneEncoding(const Image& img,
                                            int xStart, int xEnd,
                                            int upperBase, int lowerBase,
                                            bool upper) const;

    int countEndpoints  (const Image& t, int x0, int x1, int y0, int y1) const;
    int countBranches   (const Image& t, int x0, int x1, int y0, int y1) const;
    int countCrossings  (const Image& t, int x0, int x1, int y0, int y1) const;
    int countInflections(const Image& t, int x0, int x1, int y0, int y1) const;
    int countCusps      (const Image& t, int x0, int x1, int y0, int y1) const;
    int countDiacritics (const Image& t, int x0, int x1, int y0, int y1) const;
    int countLoops      (const Image& t, int x0, int x1, int y0, int y1) const;
    int countLoopsPartial(const Image& t, int x0, int x1, int y0, int y1) const;
};

// ────────────────────────────────────────────────────────────────────────────
// Free functions
// ────────────────────────────────────────────────────────────────────────────

Image loadPGM(const std::string& path);
bool  savePGM(const Image& img, const std::string& path);
Image makeSyntheticWordImage(int width = 120, int height = 40, unsigned seed = 42);

} // namespace ouhawr
