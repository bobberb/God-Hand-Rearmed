#pragma once

// Native road rung 3, step b: the draw list (claude/native-next.md "Step b"). A native packet builder registers the
// PS2 packets it wrote ("listed" packets: kamite's draw list keeps them by head address). The DMA chain walker marks
// where each listed packet sits in the VIF1 chain it gathers, and the VU worker hands those marks to the draw list's
// observer before and after it runs the chain.
//   Live (build 2, the default): the walker leaves a listed packet's bytes out of the chain it copies and keeps them as
//   pointers into guest RAM; the VU thread runs the chain up to the packet, the packet's bytes read in place, then on
//   (PS2Memory::processVIF1Listed). Exact: build 1's shadow found, over K's 36 replays (1.76M packets, Nov 24), every
//   packet's VIF commands on its own edges and its bytes in RAM unchanged until the VU job ran.
//   Shadow (dev, KAMITE_NATIVE_DRAWLIST=shadow): the walker copies everything and marks the packets; the observer checks.
// Nothing here changes what the VU or the GS receive. Small on purpose: ps2_memory.h, ps2_dma_chain.h and
// ps2_async_vif.h include it.

#include <atomic>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ps2x::listed
{
    // The VIF executor's fallback count (PS2Memory::runListedPacket): listed packets whose rest had to be gathered
    // (a command split across pieces otherwise than VIFcode | payload). Dev reports print it.
    inline std::atomic<uint64_t> executorGathers{0}; // shared-state: written and read in the program only (vif1 interpreter, draw_list)

    // A run of guest bytes (inside the RAM backing), in chain order.
    struct Piece
    {
        const uint8_t *src;
        uint32_t len;
    };

    // One listed packet in a gathered chain.
    struct Segment
    {
        uint32_t head;       // the packet's first tag (the registry's key)
        uint32_t start, end; // byte offsets in the gathered chain (live: start = end, where its bytes belong)
        uint32_t pieceBegin, pieceEnd; // its bytes as guest RAM, in Segments::pieces
        bool closed;         // false: the chain ended before its tail tag (shadow: end = start, no pieces; live: the
                             // bytes up to the chain's end)
        bool live = false;   // its bytes are not in the gathered chain (else a shadow packet: copied, and checked)
    };

    struct Segments
    {
        std::vector<Segment> segments;
        std::vector<Piece> pieces;
        bool live = false; // some packet's bytes are not in the gathered chain (Segment::live): read them from pieces
        bool empty() const { return segments.empty(); }
        void clear()
        {
            segments.clear();
            pieces.clear();
            live = false;
        }
        size_t listedBytes() const // the live packets' bytes (not in the gathered chain)
        {
            size_t n = 0;
            for (const Segment &s : segments)
                if (s.live)
                    for (uint32_t i = s.pieceBegin; i < s.pieceEnd; ++i)
                        n += pieces[i].len;
            return n;
        }
    };

    // The walker's view of a listed packet while it walks the chain.
    struct Mark
    {
        uint32_t head, tail; // first and last tag
        uint32_t start;      // byte offset in the chain before the head tag's bytes
        uint32_t end;        // after the tail tag's bytes; 0 while open
        bool closed;
        uint32_t pieceBegin = 0u, pieceEnd = 0u; // live: its bytes, in the walker's listed pieces
        bool live = false;                       // its bytes went to the listed pieces (a live entry in a live walk)
    };

    // The walker's memory of listed packets (step b's cached decode): for each head, every tag the walk visited from the
    // head to the tail (RAM offset, value), the pieces those tags produced (RAM offset, length) and the walk's state
    // after the tail. The pieces and the state depend on nothing but those tag values, so when every recorded tag still
    // holds its value the walker replays the pieces and jumps to the tail's link without walking the tags. Kept by the
    // thread that starts the DMA (PS2Memory's VIF1 kick).
    struct CacheEntry
    {
        std::vector<std::pair<uint32_t, uint64_t>> tags;   // head first, tail last
        std::vector<std::pair<uint32_t, uint32_t>> pieces; // as the walk added them (merged)
        uint32_t next = 0u, lastTagUpper = 0u, asr0 = 0u, asr1 = 0u, headAsp = 0u;
        int tagCount = 0;
        bool wrote0 = false, wrote1 = false, ready = false;
        bool tie = false, tagData = false; // the walk's CHCR.TIE / TTE when recorded
    };
    struct Cache
    {
        std::unordered_map<uint32_t, CacheEntry> entries; // by the head's RAM offset
        uint64_t hits = 0u, misses = 0u;
    };

    // The draw list's registry: is the tag at headAddr (headTag = its first 8 bytes as the walker read them) the head of
    // a listed packet, and where is its last tag? Asked for the chain's first tag and for every NEXT tag's target, on the
    // thread that starts the VIF1 DMA (the game thread). nullptr (the default, and always in releases): no marks.
    // shadow: the packet is listed for the shadow check only (a source not proven yet): the walker copies its bytes.
    using Lookup = bool (*)(uint32_t headAddr, uint64_t headTag, uint32_t &tailAddr, bool &shadow);
    // live: leave the listed packets' bytes out of the gathered chain (else: copy everything, mark only)
    void setLookup(Lookup lookup, bool live);
    Lookup lookup();
    bool live();
    // The cached decode (on by default with a lookup); its hits / misses so far (read on the game thread).
    void setCache(bool on);
    void cacheCounts(uint64_t &hits, uint64_t &misses);

    // The marks of one walk as Segments: each mark's bytes cut out of the walk's pieces (pieces = the walk's, total
    // in chain order; Piece mirrors ps2x::dma::ChainPiece). An open mark (the chain ended inside it): closed = false,
    // no pieces.
    template <class ChainPiece>
    void cutSegments(const std::vector<ChainPiece> &pieces, const std::vector<Mark> &marks, Segments &out)
    {
        out.clear();
        size_t p = 0;
        uint64_t pieceStart = 0; // chain offset of pieces[p]
        for (const Mark &m : marks)
        {
            Segment s{m.head, m.start, m.closed ? m.end : m.start, static_cast<uint32_t>(out.pieces.size()), 0u, m.closed};
            if (m.closed && m.end > m.start)
            {
                while (p < pieces.size() && pieceStart + pieces[p].len <= m.start)
                    pieceStart += pieces[p++].len;
                size_t q = p;
                uint64_t qStart = pieceStart;
                while (q < pieces.size() && qStart < m.end)
                {
                    const uint64_t from = qStart < m.start ? m.start - qStart : 0u;
                    const uint64_t to = qStart + pieces[q].len > m.end ? m.end - qStart : pieces[q].len;
                    out.pieces.push_back(Piece{pieces[q].src + from, static_cast<uint32_t>(to - from)});
                    qStart += pieces[q++].len;
                }
            }
            s.pieceEnd = static_cast<uint32_t>(out.pieces.size());
            out.segments.push_back(s);
        }
    }

    // A live walk's marks as Segments: a live mark's bytes are the walker's listed pieces (start = end, where they
    // belong), a shadow mark's are cut out of the gathered chain's pieces as cutSegments does (ChainPiece as above).
    template <class ChainPiece>
    void takeLive(const std::vector<ChainPiece> &pieces, const std::vector<ChainPiece> &listedPieces,
                  const std::vector<Mark> &marks, Segments &out)
    {
        out.clear();
        size_t p = 0;
        uint64_t pieceStart = 0;
        for (const Mark &m : marks)
        {
            if (m.live)
            {
                Segment s{m.head, m.start, m.start, static_cast<uint32_t>(out.pieces.size()), 0u, m.closed, true};
                for (uint32_t i = m.pieceBegin; i < m.pieceEnd && i < listedPieces.size(); ++i)
                    out.pieces.push_back(Piece{listedPieces[i].src, static_cast<uint32_t>(listedPieces[i].len)});
                s.pieceEnd = static_cast<uint32_t>(out.pieces.size());
                out.segments.push_back(s);
                out.live = true;
                continue;
            }
            Segment s{m.head, m.start, m.closed ? m.end : m.start, static_cast<uint32_t>(out.pieces.size()), 0u, m.closed};
            if (m.closed && m.end > m.start)
            {
                while (p < pieces.size() && pieceStart + pieces[p].len <= m.start)
                    pieceStart += pieces[p++].len;
                size_t q = p;
                uint64_t qStart = pieceStart;
                while (q < pieces.size() && qStart < m.end)
                {
                    const uint64_t from = qStart < m.start ? m.start - qStart : 0u;
                    const uint64_t to = qStart + pieces[q].len > m.end ? m.end - qStart : pieces[q].len;
                    out.pieces.push_back(Piece{pieces[q].src + from, static_cast<uint32_t>(to - from)});
                    qStart += pieces[q++].len;
                }
            }
            s.pieceEnd = static_cast<uint32_t>(out.pieces.size());
            out.segments.push_back(s);
        }
    }

    // Live: the whole chain in order as pieces (the gathered chain's pieces with each listed packet's put back where it
    // belongs), for tools that want every byte (the chain dump).
    template <class ChainPiece>
    void logicalPieces(const std::vector<ChainPiece> &pieces, const Segments &listed, std::vector<ChainPiece> &out)
    {
        out.clear();
        size_t p = 0;
        uint64_t at = 0;       // chain offset of the next byte of pieces[p]
        uint32_t used = 0;     // bytes of pieces[p] already out
        for (const Segment &s : listed.segments)
        {
            if (!s.live)
                continue; // a shadow packet's bytes are in the chain's pieces already
            while (p < pieces.size() && at < s.start)
            {
                const uint32_t left = pieces[p].len - used;
                const uint32_t take = at + left <= s.start ? left : static_cast<uint32_t>(s.start - at);
                out.push_back(ChainPiece{pieces[p].src + used, take});
                at += take;
                used += take;
                if (used == pieces[p].len)
                    ++p, used = 0;
            }
            for (uint32_t i = s.pieceBegin; i < s.pieceEnd; ++i)
                out.push_back(ChainPiece{listed.pieces[i].src, listed.pieces[i].len});
        }
        for (; p < pieces.size(); ++p, used = 0)
            out.push_back(ChainPiece{pieces[p].src + used, pieces[p].len - used});
    }
}
