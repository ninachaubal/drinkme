// CPU encoder/decoder for the research archive in radix_checkpoint.py.
// No floating point conversion: every BF16/FP8 source bit is preserved.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

struct Layout {
    uint64_t rows, cols, rg, cg, blocks, regions, stride;
    uint32_t mant, exponent, count, covered;
    const uint32_t* widths;
    Layout(uint64_t r, uint64_t c, uint64_t row_group, uint64_t block_group,
           uint32_t m, uint32_t e, const uint32_t* w, uint32_t nw)
        : rows(r), cols(c), rg(row_group), cg(block_group), mant(m), exponent(e), count(nw), widths(w) {
        if (!r || !c || !rg || !cg || r > (1ULL<<32) || c > (1ULL<<24) ||
            !((m == 7 && e == 8) || (m == 3 && e == 4)) || nw < 2 || nw > 8 || w[nw-1] != e)
            throw std::runtime_error("invalid layout");
        blocks = (c+1023)/1024;
        regions = ((r+rg-1)/rg)*((blocks+cg-1)/cg);
        covered = 0; stride = 0;
        for (uint32_t j=0; j+1<nw; ++j) {
            if (w[j] < 1 || w[j] > 8) throw std::runtime_error("invalid width");
            covered += (1U<<w[j])-1;
            stride += (((1U<<w[j])+2)/4)*4;
        }
        if (covered > (1U<<e)) throw std::runtime_error("palette exceeds alphabet");
    }
    uint64_t region(uint64_t row, uint64_t block) const {
        return (row/rg)*((blocks+cg-1)/cg) + block/cg;
    }
};

static std::vector<uint16_t> inverse(const Layout& p, const uint8_t* tables) {
    const uint32_t alphabet = 1U<<p.exponent;
    std::vector<uint16_t> inv(p.regions*alphabet, p.covered);
    for (uint64_t region=0; region<p.regions; ++region) {
        uint32_t base=0, table=0;
        for (uint32_t j=0; j+1<p.count; ++j) {
            const uint32_t n=(1U<<p.widths[j])-1, padded=((n+3)/4)*4;
            for (uint32_t k=0; k<n; ++k) {
                const uint32_t exp=tables[region*p.stride+table+k];
                if (exp >= alphabet || inv[region*alphabet+exp] != p.covered)
                    throw std::runtime_error("invalid or repeated palette value");
                inv[region*alphabet+exp]=base+k;
            }
            for (uint32_t k=n; k<padded; ++k)
                if (tables[region*p.stride+table+k]) throw std::runtime_error("nonzero palette padding");
            base+=n; table+=padded;
        }
    }
    return inv;
}

template<class T> static uint64_t plan(const T* source, const Layout& p,
                                     uint8_t* tables, uint32_t* offsets) {
    const uint32_t alphabet=1U<<p.exponent;
    std::vector<uint64_t> hist(p.regions*alphabet, 0);
    for (uint64_t row=0; row<p.rows; ++row) {
        for (uint64_t block=0; block<p.blocks; ++block) {
            auto* h=hist.data()+p.region(row,block)*alphabet;
            const uint64_t stop=std::min(p.cols,(block+1)*1024);
            for (uint64_t col=block*1024; col<stop; ++col)
                ++h[(source[row*p.cols+col]>>p.mant)&(alphabet-1)];
        }
    }
    std::memset(tables,0,p.regions*p.stride);
    for (uint64_t region=0; region<p.regions; ++region) {
        std::array<uint16_t,256> order{};
        std::iota(order.begin(),order.begin()+alphabet,0);
        const auto* h=hist.data()+region*alphabet;
        std::sort(order.begin(),order.begin()+alphabet,[&](auto a,auto b){
            return h[a] != h[b] ? h[a] > h[b] : a < b;
        });
        uint32_t base=0, table=0;
        for (uint32_t j=0; j+1<p.count; ++j) {
            const uint32_t n=(1U<<p.widths[j])-1;
            for (uint32_t k=0; k<n; ++k) tables[region*p.stride+table+k]=order[base+k];
            base+=n; table+=((n+3)/4)*4;
        }
    }
    hist.clear(); hist.shrink_to_fit();
    const auto inv=inverse(p,tables);
    uint64_t total=0;
    offsets[0]=0;
    for (uint64_t row=0; row<p.rows; ++row) {
        for (uint64_t block=0; block<p.blocks; ++block) {
            std::array<uint32_t,257> ranks{};
            const auto* lookup=inv.data()+p.region(row,block)*alphabet;
            const uint64_t n=std::min(uint64_t(1024),p.cols-block*1024);
            for (uint64_t i=0; i<n; ++i)
                ++ranks[lookup[(source[row*p.cols+block*1024+i]>>p.mant)&(alphabet-1)]];
            total+=(n*(p.mant+1)+31)/32;
            uint64_t active=n;
            uint32_t base=0;
            for (uint32_t j=0; j+1<p.count; ++j) {
                total+=(active*p.widths[j]+31)/32;
                const uint32_t end=base+(1U<<p.widths[j])-1;
                for (;base<end;++base) active-=ranks[base];
            }
            total+=(active*p.exponent+31)/32;
            if (total>UINT32_MAX) throw std::runtime_error("word directory overflow");
            offsets[row*p.blocks+block+1]=total;
        }
    }
    return total;
}

static inline void put(uint32_t* output, uint32_t index, uint32_t width, uint32_t value) {
    const uint32_t bit=index*width, shift=bit%32;
    output[bit/32] |= value<<shift;
    if (shift+width>32) output[bit/32+1] |= value>>(32-shift);
}

static inline uint32_t get(const uint32_t* input, uint32_t index, uint32_t width) {
    const uint32_t bit=index*width, shift=bit%32;
    uint32_t value=input[bit/32]>>shift;
    if (shift+width>32) value |= input[bit/32+1]<<(32-shift);
    return value&((1U<<width)-1);
}

template<class T> static void encode(const T* source, const Layout& p,
                                    const uint8_t* tables, const uint32_t* offsets,
                                    uint32_t* data, uint64_t words) {
    if (offsets[0] || offsets[p.rows*p.blocks]!=words) throw std::runtime_error("directory endpoints");
    const auto inv=inverse(p,tables);
    const uint32_t expmask=(1U<<p.exponent)-1, mantmask=(1U<<p.mant)-1;
    for (uint64_t row=0; row<p.rows; ++row) {
        for (uint64_t block=0; block<p.blocks; ++block) {
            const uint64_t id=row*p.blocks+block;
            uint64_t pos=offsets[id], end=offsets[id+1];
            if (end<=pos || end>words) throw std::runtime_error("invalid block bounds");
            std::memset(data+pos,0,(end-pos)*4);
            uint32_t active=std::min(uint64_t(1024),p.cols-block*1024);
            std::array<uint16_t,1024> ranks{}, exps{};
            const auto* lookup=inv.data()+p.region(row,block)*(1U<<p.exponent);
            const uint64_t literal_words=(active*(p.mant+1)+31)/32;
            if (literal_words>end-pos) throw std::runtime_error("short literal stream");
            for (uint32_t i=0; i<active; ++i) {
                const uint32_t bits=source[row*p.cols+block*1024+i];
                const uint32_t literal=(bits&mantmask)|((bits>>(p.mant+p.exponent))<<p.mant);
                put(data+pos,i,p.mant+1,literal);
                exps[i]=(bits>>p.mant)&expmask; ranks[i]=lookup[exps[i]];
            }
            pos+=literal_words;
            uint32_t base=0;
            for (uint32_t j=0; j+1<p.count; ++j) {
                const uint32_t width=p.widths[j], escape=(1U<<width)-1;
                const uint64_t stream_words=(active*width+31)/32;
                if (stream_words>end-pos) throw std::runtime_error("short exponent stream");
                uint32_t remaining=0;
                for (uint32_t i=0; i<active; ++i) {
                    const uint32_t code=std::min(uint32_t(ranks[i])-base,escape);
                    put(data+pos,i,width,code);
                    if (code==escape) {ranks[remaining]=ranks[i];exps[remaining++]=exps[i];}
                }
                active=remaining; base+=escape; pos+=stream_words;
            }
            const uint64_t terminal=(active*p.exponent+31)/32;
            if (terminal!=end-pos) throw std::runtime_error("terminal stream mismatch");
            for (uint32_t i=0; i<active; ++i) put(data+pos,i,p.exponent,exps[i]);
        }
    }
}

template<class T> static void decode(T* output, const Layout& p, const uint8_t* tables,
                                    const uint32_t* offsets, const uint32_t* data,
                                    uint64_t words, uint64_t first, uint64_t rows) {
    if (first>p.rows || rows>p.rows-first || offsets[0] || offsets[p.rows*p.blocks]!=words)
        throw std::runtime_error("invalid decode range or directory");
    // Full palette validation is performed by the archive reader once.
    for (uint64_t row=first; row<first+rows; ++row) {
        for (uint64_t block=0; block<p.blocks; ++block) {
            const uint64_t id=row*p.blocks+block;
            uint64_t pos=offsets[id], end=offsets[id+1];
            if (end<=pos || end>words) throw std::runtime_error("invalid block bounds");
            uint32_t active=std::min(uint64_t(1024),p.cols-block*1024);
            std::array<uint16_t,1024> indices{};
            std::iota(indices.begin(),indices.begin()+active,0);
            auto* dest=output+(row-first)*p.cols+block*1024;
            const uint64_t literal_words=(active*(p.mant+1)+31)/32;
            if (literal_words>end-pos) throw std::runtime_error("short literal stream");
            for (uint32_t i=0; i<active; ++i) {
                const uint32_t literal=get(data+pos,i,p.mant+1);
                dest[i]=(literal&((1U<<p.mant)-1))|((literal>>p.mant)<<(p.mant+p.exponent));
            }
            pos+=literal_words;
            const uint8_t* palette=tables+p.region(row,block)*p.stride;
            uint32_t table=0;
            for (uint32_t j=0; j+1<p.count; ++j) {
                const uint32_t width=p.widths[j], escape=(1U<<width)-1;
                const uint64_t stream_words=(active*width+31)/32;
                if (stream_words>end-pos) throw std::runtime_error("short exponent stream");
                uint32_t remaining=0;
                for (uint32_t i=0; i<active; ++i) {
                    const uint32_t code=get(data+pos,i,width);
                    if (code==escape) indices[remaining++]=indices[i];
                    else dest[indices[i]] |= uint32_t(palette[table+code])<<p.mant;
                }
                active=remaining; pos+=stream_words; table+=((escape+3)/4)*4;
            }
            const uint64_t terminal=(active*p.exponent+31)/32;
            if (terminal!=end-pos) throw std::runtime_error("terminal stream mismatch");
            for (uint32_t i=0; i<active; ++i) dest[indices[i]] |= get(data+pos,i,p.exponent)<<p.mant;
        }
    }
}

// One C ABI entry point keeps ctypes signatures identical for all operations.
// op: 0 plan, 1 encode, 2 decode rows, 3 validate palettes.
extern "C" int radix_native(uint32_t op, void* source_or_output, uint64_t rows, uint64_t cols,
    uint64_t rg, uint64_t cg, uint32_t mant, uint32_t exponent,
    const uint32_t* widths, uint32_t count, uint8_t* palettes, uint32_t* offsets,
    uint32_t* data, uint64_t words, uint64_t first, uint64_t nrows,
    uint64_t* result, char* error, uint64_t error_size) {
    try {
        const Layout p(rows,cols,rg,cg,mant,exponent,widths,count);
        if (op==0) {
            *result=exponent==8 ? plan(static_cast<uint16_t*>(source_or_output),p,palettes,offsets)
                                : plan(static_cast<uint8_t*>(source_or_output),p,palettes,offsets);
        } else if (op==1) {
            if (exponent==8) encode(static_cast<uint16_t*>(source_or_output),p,palettes,offsets,data,words);
            else encode(static_cast<uint8_t*>(source_or_output),p,palettes,offsets,data,words);
        } else if (op==2) {
            if (exponent==8) decode(static_cast<uint16_t*>(source_or_output),p,palettes,offsets,data,words,first,nrows);
            else decode(static_cast<uint8_t*>(source_or_output),p,palettes,offsets,data,words,first,nrows);
        } else if (op==3) { inverse(p,palettes); }
        else throw std::runtime_error("unknown operation");
        return 0;
    } catch (const std::exception& e) {
        if (error_size) {std::strncpy(error,e.what(),error_size-1);error[error_size-1]=0;}
        return 1;
    }
}
