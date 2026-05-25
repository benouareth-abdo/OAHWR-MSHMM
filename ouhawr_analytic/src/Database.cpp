/**
 * @file   Database.cpp
 * @brief  IFN/ENIT database loader and evaluation utilities.
 */

#include "Database.hpp"
#include "Preprocessing.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <random>
#include <numeric>
#include <filesystem>

namespace ouhawr {

namespace fs = std::filesystem;

// ═══════════════════════════════════════════════════════════════════════════
//  IFNENITLoader
// ═══════════════════════════════════════════════════════════════════════════

static std::vector<WordRecord> parseGroundTruth(const std::string& xmlPath,
                                                 const std::string& setId,
                                                 const std::string& rootPath)
{
    std::vector<WordRecord> records;
    std::ifstream f(xmlPath);
    if (!f) {
        // Fallback: look for .truth or .dat text files
        std::ifstream tf(rootPath + "/truth/" + setId + ".txt");
        if (!tf) return records;
        std::string line;
        while (std::getline(tf, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            WordRecord r;
            std::string imgFile;
            iss >> imgFile >> r.label;
            if (r.label.empty()) continue;
            r.imagePath = rootPath + "/sets/" + setId + "/" + imgFile;
            r.writer    = setId;
            r.partition = (setId == "e" ||setId == "f" || setId == "s" ) ? "test" : "train";
            records.push_back(r);
        }
        return records;
    }

    // Simple XML parsing (IFN/ENIT format)
    std::string line;
    while (std::getline(f, line)) {
        // Look for lines like: <Word File="..." Value="..."/>
        auto filePos = line.find("File=\"");
        auto valPos  = line.find("Value=\"");
        if (filePos == std::string::npos || valPos == std::string::npos) continue;

        filePos += 6;
        auto fileEnd = line.find('"', filePos);
        std::string imgFile = line.substr(filePos, fileEnd - filePos);

        valPos += 7;
        auto valEnd = line.find('"', valPos);
        std::string label = line.substr(valPos, valEnd - valPos);

        WordRecord r;
        r.label     = label;
        r.imagePath = rootPath + "/sets/" + setId + "/" + imgFile;
        r.writer    = setId;
        r.partition =  (setId == "e" ||setId == "f" || setId == "s" ) ? "test" : "train";
        records.push_back(r);
    }
    return records;
}

std::vector<WordRecord> IFNENITLoader::loadSet(const std::string& setId) const {
    std::string xmlPath = root_ + "/truth/" + setId + ".xml";
    return parseGroundTruth(xmlPath, setId, root_);
}

std::vector<WordRecord> IFNENITLoader::loadAll() const {
    std::vector<WordRecord> all;
    for (const char* s : {"a","b","c","d","e", "f","s"}) {
        auto v = loadSet(s);
        all.insert(all.end(), v.begin(), v.end());
    }
    return all;
}

std::vector<WordRecord> IFNENITLoader::loadTrain() const {
    std::vector<WordRecord> train;
    for (const char* s : {"a","b","c","d"}) {
        auto v = loadSet(s);
        train.insert(train.end(), v.begin(), v.end());
    }
    return train;
}

std::vector<WordRecord> IFNENITLoader::loadTest() const {
    
  std::vector<WordRecord> test;
    for (const char* s : {"e","f","s"}) {
        auto v = loadSet(s);
        test.insert(test.end(), v.begin(), v.end());
    }
    return test;
}

Image IFNENITLoader::loadImage(const WordRecord& rec) const {
    // Try PGM
    Image img = loadPGM(rec.imagePath);
    if (img.width > 0) return img;

    // Try replacing extension
    std::string path = rec.imagePath;
    auto dot = path.rfind('.');
    if (dot != std::string::npos) {
        for (const char* ext : {".pgm", ".PGM", ".png", ".bmp"}) {
            std::string p = path.substr(0, dot) + ext;
            img = loadPGM(p);
            if (img.width > 0) return img;
        }
    }
    return img;
}

std::map<std::string, std::vector<Image>>
IFNENITLoader::buildCorpus(const std::vector<WordRecord>& records) const {
    std::map<std::string, std::vector<Image>> corpus;
    for (auto& r : records) {
        Image img = loadImage(r);
        if (img.width > 0) corpus[r.label].push_back(img);
    }
    return corpus;
}

std::map<std::string, std::vector<std::string>>
IFNENITLoader::buildLexicon(const std::vector<WordRecord>& records) {
    std::map<std::string, std::vector<std::string>> lexicon;
    for (auto& r : records) {
        if (lexicon.count(r.label)) continue;
        // Build character allograph sequence from the word label
        // Each UTF-8 character is treated as one allograph (simplified)
        std::vector<std::string> chars;
        const std::string& w = r.label;
        size_t i = 0;
        while (i < w.size()) {
            unsigned char c = w[i];
            int len = 1;
            if ((c & 0x80) == 0)       len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            chars.push_back(w.substr(i, len));
            i += len;
        }
        lexicon[r.label] = chars;
    }
    return lexicon;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Evaluator
// ═══════════════════════════════════════════════════════════════════════════

EvalResult Evaluator::evaluate(const std::vector<WordRecord>& records,
                                const std::vector<std::string>& predicted)
{
    EvalResult res;
    res.total = (int)records.size();
    for (int i = 0; i < (int)records.size() && i < (int)predicted.size(); ++i)
        if (records[i].label == predicted[i]) ++res.correct;
    res.accuracy = (res.total > 0) ? 100.0 * res.correct / res.total : 0.0;
    return res;
}

void Evaluator::printReport(const EvalResult& res, const std::string& tag) {
    std::string hdr = tag.empty() ? "Evaluation" : tag;
    std::cout << "══════════════════════════════════════\n";
    std::cout << "  " << hdr << "\n";
    std::cout << "──────────────────────────────────────\n";
    std::cout << "  Total:    " << res.total   << "\n";
    std::cout << "  Correct:  " << res.correct << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Accuracy: " << res.accuracy << " %\n";
    std::cout << "══════════════════════════════════════\n";
}

std::map<std::string, double>
Evaluator::perClassAccuracy(const std::vector<WordRecord>& records,
                             const std::vector<std::string>& predicted)
{
    std::map<std::string, int> total, correct;
    for (int i = 0; i < (int)records.size() && i < (int)predicted.size(); ++i) {
        total[records[i].label]++;
        if (records[i].label == predicted[i]) correct[records[i].label]++;
    }
    std::map<std::string, double> acc;
    for (auto& [label, cnt] : total)
        acc[label] = 100.0 * correct[label] / cnt;
    return acc;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Synthetic dataset
// ═══════════════════════════════════════════════════════════════════════════

void makeSyntheticDataset(
        int numClasses,
        int numSamplesPerClass,
        int numTestSamples,
        std::map<std::string, std::vector<Image>>& corpus,
        std::map<std::string, std::vector<Image>>& testCorpus,
        std::map<std::string, std::vector<std::string>>& lexicon,
        unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> noiseDist(0, 20);
    std::uniform_int_distribution<> strokeDist(2, 6);

    corpus.clear(); testCorpus.clear(); lexicon.clear();

    for (int cls = 0; cls < numClasses; ++cls) {
        // Create class label (simulate Arabic-like word labels)
        std::string label = "word_" + std::to_string(cls);

        // Character allograph sequence (2-4 chars per word)
        int numChars = 2 + (int)(rng() % 3);
        std::vector<std::string> chars;
        for (int k = 0; k < numChars; ++k)
            chars.push_back("char_" + std::to_string((cls * 7 + k) % 20));
        lexicon[label] = chars;

        // Prototype image (60×30, with class-specific pattern)
        int W = 60 + (int)(rng() % 40), H = 30;
        Image proto(W, H);
        // Draw a class-specific pattern: horizontal strokes at class-dependent rows
        int baseRow = H/4 + (cls % 3) * (H/6);
        for (int x = 5; x < W-5; x += 3) proto.at(x, baseRow) = 255;
        for (int x = 3; x < W/2; x += 2)  proto.at(x, baseRow+2) = 255;
        for (int x = W/2; x < W-3; x += 4) proto.at(x, baseRow-1) = 255;
        // Vertical stroke
        for (int y = H/4; y < 3*H/4; ++y) proto.at(W/3, y) = 255;

        // Generate training samples (noisy variants)
        for (int s = 0; s < numSamplesPerClass; ++s) {
            Image img = proto;
            // Add salt & pepper noise
            for (int x = 0; x < W; ++x)
                for (int y = 0; y < H; ++y)
                    if (noiseDist(rng) == 0) img.at(x,y) = img.isForeground(x,y) ? 0 : 255;
            corpus[label].push_back(img);
        }

        // Generate test samples
        for (int s = 0; s < numTestSamples; ++s) {
            Image img = proto;
            for (int x = 0; x < W; ++x)
                for (int y = 0; y < H; ++y)
                    if (noiseDist(rng) < 2) img.at(x,y) = img.isForeground(x,y) ? 0 : 255;
            testCorpus[label].push_back(img);
        }
    }
}

} // namespace ouhawr
