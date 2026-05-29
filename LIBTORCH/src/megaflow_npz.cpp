// ============================================================================
// megaflow_npz.cpp  —  byte-compatible np.savez writer for saveCache.
//
// Reproduces megaflow_cache.py::save_cache:
//   np.savez(path, traj_maps=<fp16 (T,2,H,W)>, meta=np.array(json.dumps(meta)))
// np.savez writes an UNCOMPRESSED (ZIP_STORED) archive of two .npy members:
//   traj_maps.npy : dtype '<f2', shape (T,2,H,W), C-order
//   meta.npy      : dtype '<U{N}', shape ()  (0-d), data = JSON as UTF-32-LE
// load_cache / diff_npz read these back via np.load + json.loads(str(meta)).
//
// Hand-rolled (no cnpy dep) so the meta '<U' 0-d string is exact — that's the
// part a generic npy lib gets wrong.
// ============================================================================
#include "megaflow_solve.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace megaflow {

// --- CRC32 (zip polynomial 0xEDB88320) ---------------------------------------
static uint32_t crc32_buf(const std::string& s)
{
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (unsigned char ch : s) crc = table[(crc ^ ch) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

// --- little-endian append helpers --------------------------------------------
static void put_u16(std::string& o, uint16_t v) {
    o.push_back((char)(v & 0xFF)); o.push_back((char)((v >> 8) & 0xFF));
}
static void put_u32(std::string& o, uint32_t v) {
    for (int i = 0; i < 4; ++i) o.push_back((char)((v >> (8 * i)) & 0xFF));
}

// --- .npy v1.0 header (descr + shape), 64-byte aligned ------------------------
static std::string npyHeader(const std::string& descr, const std::vector<int64_t>& shape)
{
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) sh += ", ";
        sh += std::to_string(shape[i]);
    }
    if (shape.size() == 1) sh += ",";
    sh += ")";

    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t total = 10 + dict.size() + 1;            // 6 magic + 2 ver + 2 len + dict + '\n'
    size_t pad = (64 - (total % 64)) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');

    std::string out;
    out.push_back((char)0x93); out += "NUMPY";
    out.push_back('\x01'); out.push_back('\x00');   // version 1.0
    put_u16(out, (uint16_t)dict.size());
    out += dict;
    return out;
}

// --- traj_maps.npy : fp16 (T,2,H,W) ------------------------------------------
static std::string npyFloat16(const torch::Tensor& t_in)
{
    auto t = t_in.to(torch::kCPU).to(torch::kHalf).contiguous();
    std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
    std::string out = npyHeader("<f2", shape);
    out.append(reinterpret_cast<const char*>(t.data_ptr()),
               (size_t)t.numel() * 2);              // 2 bytes / fp16 element
    return out;
}

// --- meta.npy : 0-d '<U{N}' unicode string, data = UTF-32-LE codepoints -------
static std::string npyMetaString(const std::string& s)
{
    std::string out = npyHeader("<U" + std::to_string(s.size()), {});   // shape ()
    for (unsigned char c : s) {                      // ASCII (json.dumps ensure_ascii=True)
        uint32_t cp = c;
        out.push_back((char)(cp & 0xFF));
        out.push_back((char)((cp >> 8) & 0xFF));
        out.push_back((char)((cp >> 16) & 0xFF));
        out.push_back((char)((cp >> 24) & 0xFF));
    }
    return out;
}

// --- STORED zip of named .npy members ----------------------------------------
static void writeNpzStored(const std::string& path,
                           const std::vector<std::pair<std::string, std::string>>& entries)
{
    std::string out;
    struct CD { std::string name; uint32_t crc, size, off; };
    std::vector<CD> cds;

    for (const auto& e : entries) {
        const std::string& name = e.first;
        const std::string& data = e.second;
        uint32_t crc = crc32_buf(data);
        uint32_t off = (uint32_t)out.size();

        out += "PK\x03\x04";
        put_u16(out, 20); put_u16(out, 0); put_u16(out, 0);   // ver, flags, comp=stored
        put_u16(out, 0);  put_u16(out, 0);                     // modtime, moddate
        put_u32(out, crc); put_u32(out, (uint32_t)data.size()); put_u32(out, (uint32_t)data.size());
        put_u16(out, (uint16_t)name.size()); put_u16(out, 0);  // name len, extra len
        out += name; out += data;

        cds.push_back({name, crc, (uint32_t)data.size(), off});
    }

    uint32_t cd_start = (uint32_t)out.size();
    for (const auto& c : cds) {
        out += "PK\x01\x02";
        put_u16(out, 20); put_u16(out, 20);                    // ver made by, ver needed
        put_u16(out, 0); put_u16(out, 0); put_u16(out, 0); put_u16(out, 0);  // flags,comp,modt,modd
        put_u32(out, c.crc); put_u32(out, c.size); put_u32(out, c.size);
        put_u16(out, (uint16_t)c.name.size());
        put_u16(out, 0); put_u16(out, 0);                      // extra, comment len
        put_u16(out, 0); put_u16(out, 0);                      // disk start, internal attrs
        put_u32(out, 0); put_u32(out, c.off);                  // external attrs, local hdr offset
        out += c.name;
    }
    uint32_t cd_size = (uint32_t)out.size() - cd_start;

    out += "PK\x05\x06";
    put_u16(out, 0); put_u16(out, 0);
    put_u16(out, (uint16_t)cds.size()); put_u16(out, (uint16_t)cds.size());
    put_u32(out, cd_size); put_u32(out, cd_start);
    put_u16(out, 0);

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("saveCache: cannot open '" + path + "' for writing");
    f.write(out.data(), (std::streamsize)out.size());
    if (!f) throw std::runtime_error("saveCache: write failed for '" + path + "'");
}

void saveCache(const std::string& path, const torch::Tensor& traj_maps,
               const std::string& meta_json)
{
    // mirror save_cache's os.makedirs(out_dir, exist_ok=True)
    try {
        auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
    } catch (const std::exception&) { /* fall through; the open will report */ }

    writeNpzStored(path, {
        {"traj_maps.npy", npyFloat16(traj_maps)},
        {"meta.npy",      npyMetaString(meta_json)},
    });
}

} // namespace megaflow
