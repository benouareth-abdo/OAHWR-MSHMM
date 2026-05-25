/**
 * @file   Preprocessing.cpp
 * @brief  Implementation of Preprocessor (§5.1) and FeatureExtractor (§5.2).
 *
 * Four feature streams per sliding-window frame (right-to-left scan):
 *   Stream 0 – upper-contour  (15-dim): 8 direction-histogram bins + 7 zone/category
 *   Stream 1 – lower-contour  (15-dim): idem for lower contour
 *   Stream 2 – statistical    (26-dim): density, projection, concavity features
 *   Stream 3 – structural     (24-dim): 3 zones × 8 skeleton feature types
 */

#include "Preprocessing.hpp"
#include <fstream>
#include <sstream>
#include <cstring>
#include <queue>
#include <random>

namespace ouhawr {

// ═══════════════════════════════════════════════════════════════════════════
//  Preprocessor – §5.1
// ═══════════════════════════════════════════════════════════════════════════

Image Preprocessor::process(const Image& src) const {
    Image smoothed = smooth(src);
    return thin(smoothed);
}

// ── Step 1: Box smoothing ──────────────────────────────────────────────────

Image Preprocessor::smooth(const Image& src) const {
    Image out(src.width, src.height);
    int r = params_.smoothRadius;
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            int sum = 0, cnt = 0;
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    int nx = x + dx, ny = y + dy;
                    if (src.valid(nx, ny)) { sum += src.at(nx, ny); ++cnt; }
                }
            }
            out.at(x, y) = cnt ? static_cast<uint8_t>(sum / cnt) : 0;
        }
    }
    return out;
}

// ── Step 4: Pavlidis thinning ─────────────────────────────────────────────

static int neighbourCount(const Image& img, int x, int y) {
    int cnt = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if ((dx || dy) && img.valid(x+dx, y+dy) && img.isForeground(x+dx, y+dy))
                ++cnt;
    return cnt;
}

static int transitions(const Image& img, int x, int y) {
    static const int cx[8] = {0,1,1,1,0,-1,-1,-1};
    static const int cy[8] = {-1,-1,0,1,1,1,0,-1};
    int t = 0;
    for (int i = 0; i < 8; ++i) {
        bool a = img.valid(x+cx[i],   y+cy[i])   && img.isForeground(x+cx[i],   y+cy[i]);
        bool b = img.valid(x+cx[(i+1)%8], y+cy[(i+1)%8]) && img.isForeground(x+cx[(i+1)%8], y+cy[(i+1)%8]);
        if (!a && b) ++t;
    }
    return t;
}

bool Preprocessor::canRemove(const Image& img, int x, int y) const {
    int nc = neighbourCount(img, x, y);
    if (nc < 2 || nc > 6) return false;
    return transitions(img, x, y) == 1;
}

Image Preprocessor::thin(const Image& src) const {
    Image cur = src;
    for (int iter = 0; iter < params_.thinningIter; ++iter) {
        bool changed = false;
        // Sub-iteration 1: remove north/south border pixels
        Image next = cur;
        for (int y = 1; y < cur.height-1; ++y) {
            for (int x = 1; x < cur.width-1; ++x) {
                if (!cur.isForeground(x,y)) continue;
                bool pN = cur.isForeground(x, y-1);
                bool pS = cur.isForeground(x, y+1);
                bool pE = cur.isForeground(x+1, y);
                bool pW = cur.isForeground(x-1, y);
                if (canRemove(cur,x,y) && !(pN && pS) && !(pE && pW)) {
                    next.at(x,y) = 0; changed = true;
                }
            }
        }
        cur = next;
        // Sub-iteration 2: remove east/west border pixels
        for (int y = 1; y < cur.height-1; ++y) {
            for (int x = 1; x < cur.width-1; ++x) {
                if (!cur.isForeground(x,y)) continue;
                bool pN = cur.isForeground(x, y-1);
                bool pS = cur.isForeground(x, y+1);
                bool pE = cur.isForeground(x+1, y);
                bool pW = cur.isForeground(x-1, y);
                if (canRemove(cur,x,y) && !(pN && pE) && !(pS && pW)) {
                    next.at(x,y) = 0; changed = true;
                }
            }
        }
        cur = next;
        if (!changed) break;
    }
    return cur;
}

// ── Step 3: Baseline estimation ───────────────────────────────────────────

void Preprocessor::estimateBaselines(const Image& img, int& upper, int& lower) const {
    std::vector<int> rowCount(img.height, 0);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x)
            if (img.isForeground(x,y)) ++rowCount[y];

    int H = img.height;
    int upperH = static_cast<int>(params_.upperZoneFraction * H);
    int lowerH = static_cast<int>(params_.lowerZoneFraction * H);

    upper = upperH;
    for (int y = 0; y < upperH; ++y)
        if (rowCount[y] > img.width * 0.05) { upper = y; break; }

    lower = H - lowerH;
    for (int y = H-1; y >= H-lowerH; --y)
        if (rowCount[y] > img.width * 0.05) { lower = y; break; }

    // Guard
    if (upper >= lower) { upper = H/3; lower = 2*H/3; }
}

// ── Step 2: Freeman 8-connected chain code ────────────────────────────────

std::vector<int> Preprocessor::chainCode(const Image& img) const {
    // Find start pixel: rightmost foreground pixel (Arabic is right-to-left)
    int startX = -1, startY = -1;
    for (int x = img.width-1; x >= 0 && startX < 0; --x)
        for (int y = 0; y < img.height && startX < 0; ++y)
            if (img.isForeground(x,y)) { startX=x; startY=y; }
    if (startX < 0) return {};

    std::vector<int> cc;
    int cx = startX, cy = startY;
    int prevDir = 0;
    std::vector<std::vector<bool>> visited(img.height, std::vector<bool>(img.width, false));

    for (int step = 0; step < img.width * img.height; ++step) {
        visited[cy][cx] = true;
        bool found = false;
        for (int k = 0; k < 8; ++k) {
            int d = (prevDir + 7 - 1 + k) % 8;
            int nx = cx + DX8[d], ny = cy + DY8[d];
            if (img.valid(nx,ny) && img.isForeground(nx,ny) && !visited[ny][nx]) {
                cc.push_back(d);
                cx = nx; cy = ny; prevDir = d; found = true; break;
            }
        }
        if (!found || (cx == startX && cy == startY)) break;
    }
    return cc;
}

bool Preprocessor::isContourPixel(const Image& img, int x, int y) const {
    if (!img.isForeground(x,y)) return false;
    for (int dy=-1; dy<=1; ++dy)
        for (int dx=-1; dx<=1; ++dx)
            if ((dx||dy) && img.valid(x+dx,y+dy) && !img.isForeground(x+dx,y+dy))
                return true;
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
//  FeatureExtractor – §5.2
// ═══════════════════════════════════════════════════════════════════════════

ObsSequence FeatureExtractor::extract(const Image& raw) const {
    // 1. Preprocess: smooth + thin
    Image smoothed = prep_.smooth(raw);
    Image thinned  = prep_.thin(smoothed);

    // 2. Estimate baselines on the smoothed image
    int upperBase, lowerBase;
    prep_.estimateBaselines(smoothed, upperBase, lowerBase);

    // 3. Chain code on the smoothed image (§5.1)
    std::vector<int> cc = prep_.chainCode(smoothed);

    // 4. Sliding window: right-to-left (§5.2)
    ObsSequence seq;
    int W     = smoothed.width;
    int step  = prep_.params().frameStep;
    int fW    = prep_.params().frameWidth;

    for (int xEnd = W-1; xEnd >= 0; xEnd -= step) {
        int xStart = std::max(0, xEnd - fW + 1);

        Frame f;
        f.upperContour = extractContourFeatures(smoothed, cc, xStart, xEnd,
                                                upperBase, lowerBase, true);
        f.lowerContour = extractContourFeatures(smoothed, cc, xStart, xEnd,
                                                upperBase, lowerBase, false);
        f.statistical  = extractStatistical(smoothed, xStart, xEnd, upperBase, lowerBase);
        f.structural   = extractStructural(thinned, xStart, xEnd, upperBase, lowerBase);
        seq.push_back(std::move(f));
    }
    return seq;
}

// ── Stream 0 & 1: contour features – 15-dim (§5.2.1) ────────────────────
// 8 Freeman direction histogram + 4 endpoint categories + 3 zone locations

std::vector<double> FeatureExtractor::extractContourFeatures(
        const Image& proc, const std::vector<int>& cc,
        int xStart, int xEnd, int upperBase, int lowerBase, bool upper) const
{
    std::vector<double> feat(15, 0.0);

    // 8 direction histogram bins
    std::vector<double> dirHist = contourDirectionHistogram(cc, xStart, xEnd);
    std::copy(dirHist.begin(), dirHist.end(), feat.begin());

    // 4 endpoint categories + 3 zone locations  (indices 8–14)
    std::vector<double> zoneEnc = contourZoneEncoding(proc, xStart, xEnd,
                                                       upperBase, lowerBase, upper);
    for (int i = 0; i < 7 && i < (int)zoneEnc.size(); ++i)
        feat[8 + i] = zoneEnc[i];

    return feat;
}

std::vector<double> FeatureExtractor::contourDirectionHistogram(
        const std::vector<int>& cc, int /*xStart*/, int /*xEnd*/) const
{
    std::vector<double> hist(8, 0.0);
    for (int d : cc) if (d >= 0 && d < 8) hist[d] += 1.0;
    double total = std::accumulate(hist.begin(), hist.end(), 0.0);
    if (total > 0) for (auto& h : hist) h /= total;
    return hist;
}

std::vector<double> FeatureExtractor::contourZoneEncoding(
        const Image& img, int xStart, int xEnd,
        int upperBase, int lowerBase, bool upper) const
{
    // 4 categories: lower-contour(0), interior/loop(1), upper-contour(2), not-found(3)
    // 3 zone locations: upper zone(4), middle zone(5), lower zone(6)
    std::vector<double> feat(7, 0.0);
    int cnt = 0;
    for (int x = xStart; x <= xEnd && x < img.width; ++x) {
        int contourY = -1;
        if (upper) {
            for (int y = 0; y < img.height; ++y)
                if (img.isForeground(x,y)) { contourY = y; break; }
        } else {
            for (int y = img.height-1; y >= 0; --y)
                if (img.isForeground(x,y)) { contourY = y; break; }
        }
        if (contourY < 0) { feat[3] += 1.0; ++cnt; continue; }

        // Category classification
        if (contourY <= upperBase)      feat[2] += 1.0;  // upper contour
        else if (contourY >= lowerBase) feat[0] += 1.0;  // lower contour
        else                            feat[1] += 1.0;  // interior

        // Zone location
        if (contourY < upperBase)        feat[4] += 1.0;
        else if (contourY >= lowerBase)  feat[6] += 1.0;
        else                             feat[5] += 1.0;
        ++cnt;
    }
    if (cnt > 0) for (auto& f : feat) f /= cnt;
    return feat;
}

// ── Stream 2: Statistical features – 26-dim (§5.2.2.1) ──────────────────

std::vector<double> FeatureExtractor::extractStatistical(
        const Image& proc, int xStart, int xEnd,
        int upperBase, int lowerBase) const
{
    std::vector<double> feat(26, 0.0);
    int W = std::max(1, xEnd - xStart + 1);
    int H = proc.height;

    // Vertical projection: foreground pixel count per column
    std::vector<int> vproj(W, 0);
    for (int xi = 0; xi < W; ++xi) {
        int x = xEnd - xi;
        if (x < 0 || x >= proc.width) continue;
        for (int y = 0; y < H; ++y)
            if (proc.isForeground(x,y)) ++vproj[xi];
    }

    // 0–3: mean, std, min, max of vertical projection (normalised)
    double vpMean = 0, vpVar = 0, vpMin = H, vpMax = 0;
    for (int v : vproj) {
        vpMean += v;
        vpMin = std::min(vpMin, static_cast<double>(v));
        vpMax = std::max(vpMax, static_cast<double>(v));
    }
    vpMean /= W;
    for (int v : vproj) vpVar += (v-vpMean)*(v-vpMean);
    vpVar /= W;
    feat[0] = vpMean/H; feat[1] = std::sqrt(vpVar)/H;
    feat[2] = vpMin /H; feat[3] = vpMax /H;

    // 4–7: mean, std, min, max of transition count per column
    std::vector<int> trans(W, 0);
    for (int xi = 0; xi < W; ++xi) {
        int x = xEnd - xi;
        if (x < 0 || x >= proc.width) continue;
        for (int y = 1; y < H; ++y)
            if (proc.isForeground(x,y-1) != proc.isForeground(x,y)) ++trans[xi];
    }
    double tMean = 0, tVar = 0, tMin = H*2.0, tMax = 0;
    for (int t : trans) {
        tMean += t;
        tMin = std::min(tMin, static_cast<double>(t));
        tMax = std::max(tMax, static_cast<double>(t));
    }
    tMean /= W;
    for (int t : trans) tVar += (t-tMean)*(t-tMean);
    tVar /= W;
    feat[4] = tMean/H; feat[5] = std::sqrt(tVar)/H;
    feat[6] = tMin /H; feat[7] = tMax /H;

    // 8: overall pixel density
    int totalFG = 0;
    for (int v : vproj) totalFG += v;
    feat[8] = static_cast<double>(totalFG) / (W * H);

    // 9–11: gravity centres (x, y, middle-zone y)
    double gcX = 0, gcY = 0, gcMid = 0; int gcCnt = 0;
    for (int xi = 0; xi < W; ++xi) {
        int x = xEnd - xi;
        if (x < 0 || x >= proc.width) continue;
        for (int y = 0; y < H; ++y) {
            if (proc.isForeground(x,y)) {
                gcX += x; gcY += y;
                if (y >= upperBase && y <= lowerBase) gcMid += y;
                ++gcCnt;
            }
        }
    }
    if (gcCnt > 0) {
        feat[9]  = gcX / (gcCnt * proc.width);
        feat[10] = gcY / (gcCnt * H);
        feat[11] = gcMid / (gcCnt * H);
    }

    // 12–14: zone density (upper, middle, lower)
    auto zoneDensity = [&](int y0, int y1) -> double {
        int cnt = 0, tot = 0;
        for (int xi = 0; xi < W; ++xi) {
            int x = xEnd - xi;
            if (x < 0 || x >= proc.width) continue;
            for (int y = y0; y < y1 && y < H; ++y) { ++tot; if (proc.isForeground(x,y)) ++cnt; }
        }
        return tot > 0 ? static_cast<double>(cnt)/tot : 0.0;
    };
    feat[12] = zoneDensity(0, upperBase);
    feat[13] = zoneDensity(upperBase, lowerBase);
    feat[14] = zoneDensity(lowerBase, H);

    // 15–17: baseline positions and middle-zone height (normalised)
    feat[15] = static_cast<double>(upperBase) / H;
    feat[16] = static_cast<double>(lowerBase) / H;
    feat[17] = static_cast<double>(lowerBase - upperBase) / H;

    // 18–19: concavity features – full frame and middle zone
    auto concavity = [&](int y0, int y1) -> double {
        int concave = 0, total = 0;
        for (int y = y0; y < y1 && y < H; ++y) {
            bool inFG = false, wasGap = false;
            for (int xi = 0; xi < W; ++xi) {
                int x = xEnd - xi;
                if (x < 0 || x >= proc.width) continue;
                bool fg = proc.isForeground(x,y);
                if (fg && wasGap) ++concave;
                wasGap = fg ? false : inFG;
                if (fg) inFG = true;
                ++total;
            }
        }
        return total > 0 ? static_cast<double>(concave)/total : 0.0;
    };
    feat[18] = concavity(0, H);
    feat[19] = concavity(upperBase, lowerBase);

    // 20–21: horizontal projection mean and std (normalised)
    double hpMean = 0, hpVar = 0;
    for (int y = 0; y < H; ++y) {
        int c = 0;
        for (int xi = 0; xi < W; ++xi) {
            int x = xEnd - xi;
            if (x >= 0 && x < proc.width && proc.isForeground(x,y)) ++c;
        }
        hpMean += c;
    }
    hpMean /= H;
    for (int y = 0; y < H; ++y) {
        int c = 0;
        for (int xi = 0; xi < W; ++xi) {
            int x = xEnd - xi;
            if (x >= 0 && x < proc.width && proc.isForeground(x,y)) ++c;
        }
        hpVar += (c - hpMean)*(c - hpMean);
    }
    feat[20] = hpMean / W;
    feat[21] = std::sqrt(hpVar / H) / W;

    // 22–25: additional baseline-relative and density descriptors
    feat[22] = feat[15];   // upper baseline fraction
    feat[23] = 1.0 - feat[16]; // distance from lower baseline to bottom
    feat[24] = feat[8];    // density (repeat for dim=26)
    feat[25] = feat[17];   // middle-zone height

    return feat;
}

// ── Stream 3: Structural features – 24-dim (§5.2.2.2) ───────────────────
// 3 zones × 8 types: endpoints, branches, crossings, inflections,
//                    cusps, diacritics, complete loops, partial loops

std::vector<double> FeatureExtractor::extractStructural(
        const Image& thinned, int xStart, int xEnd,
        int upperBase, int lowerBase) const
{
    std::vector<double> feat(24, 0.0);

    struct Zone { int y0, y1, offset; };
    Zone zones[3] = {
        {0,          upperBase,        0},
        {upperBase,  lowerBase,        7},
        {lowerBase,  thinned.height,  14}
    };

    for (auto& z : zones) {
        if (z.y0 >= z.y1) continue;
        feat[z.offset + 0] = static_cast<double>(countEndpoints  (thinned, xStart, xEnd, z.y0, z.y1));
        feat[z.offset + 1] = static_cast<double>(countBranches   (thinned, xStart, xEnd, z.y0, z.y1));
        feat[z.offset + 2] = static_cast<double>(countCrossings  (thinned, xStart, xEnd, z.y0, z.y1));
        feat[z.offset + 3] = static_cast<double>(countInflections(thinned, xStart, xEnd, z.y0, z.y1));
        feat[z.offset + 4] = static_cast<double>(countCusps      (thinned, xStart, xEnd, z.y0, z.y1));
        feat[z.offset + 5] = static_cast<double>(countDiacritics (thinned, xStart, xEnd, z.y0, z.y1));
		feat[z.offset + 6] = static_cast<double>(countLoops(thinned, xStart, xEnd, z.y0, z.y1));
        // Feature 6: total loop count (complete + partial)
        feat[z.offset + 7] = static_cast<double>(countLoopsPartial(thinned, xStart, xEnd, z.y0, z.y1));
    }

    // Normalise by maximum value for numerical stability
    double maxVal = *std::max_element(feat.begin(), feat.end());
    if (maxVal > 0) for (auto& f : feat) f /= maxVal;
    return feat;
}

// ── Skeleton feature helpers ──────────────────────────────────────────────

int FeatureExtractor::countEndpoints(const Image& img, int x0, int x1, int y0, int y1) const {
    int cnt = 0;
    for (int x = x0; x <= x1 && x < img.width; ++x)
        for (int y = y0; y < y1 && y < img.height; ++y)
            if (img.isForeground(x,y)) {
                int n = 0;
                for (int dy=-1;dy<=1;++dy)
                    for (int dx=-1;dx<=1;++dx)
                        if ((dx||dy) && img.valid(x+dx,y+dy) && img.isForeground(x+dx,y+dy)) ++n;
                if (n == 1) ++cnt;
            }
    return cnt;
}

int FeatureExtractor::countBranches(const Image& img, int x0, int x1, int y0, int y1) const {
    int cnt = 0;
    for (int x = x0; x <= x1 && x < img.width; ++x)
        for (int y = y0; y < y1 && y < img.height; ++y)
            if (img.isForeground(x,y)) {
                int n = 0;
                for (int dy=-1;dy<=1;++dy)
                    for (int dx=-1;dx<=1;++dx)
                        if ((dx||dy) && img.valid(x+dx,y+dy) && img.isForeground(x+dx,y+dy)) ++n;
                if (n >= 3) ++cnt;
            }
    return cnt;
}

int FeatureExtractor::countCrossings(const Image& img, int x0, int x1, int y0, int y1) const {
    int cnt = 0;
    for (int x = x0; x <= x1 && x < img.width; ++x)
        for (int y = y0; y < y1 && y < img.height; ++y)
            if (img.isForeground(x,y)) {
                int n = 0;
                for (int dy=-1;dy<=1;++dy)
                    for (int dx=-1;dx<=1;++dx)
                        if ((dx||dy) && img.valid(x+dx,y+dy) && img.isForeground(x+dx,y+dy)) ++n;
                if (n >= 4) ++cnt;
            }
    return cnt;
}

int FeatureExtractor::countInflections(const Image& img, int x0, int x1, int y0, int y1) const {
    // Degree-2 pixel where the two neighbours are not collinear (cross product ≠ 0)
    int cnt = 0;
    for (int x = x0; x <= x1 && x < img.width; ++x) {
        for (int y = y0; y < y1 && y < img.height; ++y) {
            if (!img.isForeground(x,y)) continue;
            std::vector<std::pair<int,int>> nb;
            for (int dy=-1;dy<=1;++dy)
                for (int dx=-1;dx<=1;++dx)
                    if ((dx||dy) && img.valid(x+dx,y+dy) && img.isForeground(x+dx,y+dy))
                        nb.push_back({dx,dy});
            if (nb.size() == 2) {
                int cross = nb[0].first*nb[1].second - nb[1].first*nb[0].second;
                if (cross != 0) ++cnt;
            }
        }
    }
    return cnt;
}

int FeatureExtractor::countCusps(const Image& img, int x0, int x1, int y0, int y1) const {
    // Degree-2 pixel with orthogonal neighbours (dot product == 0) → sharp corner
    int cnt = 0;
    for (int x = x0; x <= x1 && x < img.width; ++x) {
        for (int y = y0; y < y1 && y < img.height; ++y) {
            if (!img.isForeground(x,y)) continue;
            std::vector<std::pair<int,int>> nb;
            for (int dy=-1;dy<=1;++dy)
                for (int dx=-1;dx<=1;++dx)
                    if ((dx||dy) && img.valid(x+dx,y+dy) && img.isForeground(x+dx,y+dy))
                        nb.push_back({dx,dy});
            if (nb.size() == 2) {
                int dot = nb[0].first*nb[1].first + nb[0].second*nb[1].second;
                if (dot == 0) ++cnt;
            }
        }
    }
    return cnt;
}

int FeatureExtractor::countDiacritics(const Image& img, int x0, int x1, int y0, int y1) const {
    // Isolated foreground pixel (0 neighbours) → diacritic dot
    int cnt = 0;
    for (int x = x0; x <= x1 && x < img.width; ++x) {
        for (int y = y0; y < y1 && y < img.height; ++y) {
            if (!img.isForeground(x,y)) continue;
            int n = 0;
            for (int dy=-1;dy<=1;++dy)
                for (int dx=-1;dx<=1;++dx)
                    if ((dx||dy) && img.valid(x+dx,y+dy) && img.isForeground(x+dx,y+dy)) ++n;
            if (n == 0) ++cnt;
        }
    }
    return cnt;
}

int FeatureExtractor::countLoops(const Image& img, int x0, int x1, int y0, int y1) const {
    // Count enclosed background regions entirely inside image (not touching border)
    // that are strictly contained within the zone [y0,y1]
    std::vector<std::vector<bool>> visited(img.height, std::vector<bool>(img.width, false));
    int loops = 0;
    for (int y = y0; y < y1 && y < img.height; ++y) {
        for (int x = x0; x <= x1 && x < img.width; ++x) {
            if (img.isForeground(x,y) || visited[y][x]) continue;
            // BFS over connected background component
            bool touchesBorder = false;
            std::queue<std::pair<int,int>> q;
            q.push({x,y}); visited[y][x] = true;
            while (!q.empty()) {
                auto [cx,cy] = q.front(); q.pop();
                if (cx==0 || cx==img.width-1 || cy==0 || cy==img.height-1)
                    touchesBorder = true;
                for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx) {
                    if (!dx && !dy) continue;
                    int nx=cx+dx, ny=cy+dy;
                    if (img.valid(nx,ny) && !img.isForeground(nx,ny) && !visited[ny][nx]) {
                        visited[ny][nx] = true; q.push({nx,ny});
                    }
                }
            }
            if (!touchesBorder) ++loops;
        }
    }
    return loops;
}

int FeatureExtractor::countLoopsPartial(const Image& img, int x0, int x1, int y0, int y1) const {
    // Enclosed background components that intersect the zone but may extend outside
    std::vector<std::vector<bool>> visited(img.height, std::vector<bool>(img.width, false));
    int loops = 0;
    for (int y = y0; y < y1 && y < img.height; ++y) {
        for (int x = x0; x <= x1 && x < img.width; ++x) {
            if (img.isForeground(x,y) || visited[y][x]) continue;
            bool touchesBorder = false;
            std::queue<std::pair<int,int>> q;
            q.push({x,y}); visited[y][x] = true;
            while (!q.empty()) {
                auto [cx,cy] = q.front(); q.pop();
                if (cx==0 || cx==img.width-1 || cy==0 || cy==img.height-1)
                    touchesBorder = true;
                for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx) {
                    if (!dx && !dy) continue;
                    int nx=cx+dx, ny=cy+dy;
                    if (img.valid(nx,ny) && !img.isForeground(nx,ny) && !visited[ny][nx]) {
                        visited[ny][nx] = true; q.push({nx,ny});
                    }
                }
            }
            if (!touchesBorder) ++loops;
        }
    }
    return loops;
}

// ═══════════════════════════════════════════════════════════════════════════
//  I/O utilities
// ═══════════════════════════════════════════════════════════════════════════

Image loadPGM(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::string magic;
    f >> magic;
    if (magic != "P5" && magic != "P2") return {};
    int w, h, maxVal;
    f >> w >> h >> maxVal;
    f.ignore(1);
    Image img(w, h);
    if (magic == "P5") {
        f.read(reinterpret_cast<char*>(img.data.data()), w*h);
    } else {
        for (auto& px : img.data) { int v; f >> v; px = static_cast<uint8_t>(v); }
    }
    return img;
}

bool savePGM(const Image& img, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P5\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
    return f.good();
}

Image makeSyntheticWordImage(int width, int height, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> noise(0.0, 1.0);
    Image img(width, height);
    // Horizontal strokes simulating Arabic ligatures
    for (int s = 0; s < 3; ++s) {
        int y = static_cast<int>(height * (0.3 + 0.15 * s));
        for (int x = width/5; x < 4*width/5; ++x) {
            if (noise(rng) < 0.85) {
                img.at(x,y) = 255;
                if (y > 0         && noise(rng) < 0.4) img.at(x, y-1) = 255;
                if (y < height-1  && noise(rng) < 0.4) img.at(x, y+1) = 255;
            }
        }
    }
    // Diacritic dots
    for (int i = 0; i < 4; ++i) {
        int x = static_cast<int>(width * (0.1 + 0.2*i));
        int y = static_cast<int>(height * 0.15);
        for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx)
            if (img.valid(x+dx,y+dy)) img.at(x+dx,y+dy) = 255;
    }
    return img;
}

// constexpr static member definitions
constexpr int Preprocessor::DX8[8];
constexpr int Preprocessor::DY8[8];

} // namespace ouhawr
