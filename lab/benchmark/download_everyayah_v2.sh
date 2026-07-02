#!/bin/bash
# Download fresh EveryAyah samples for v2 validation corpus.
# Uses Husary reciter (different from v1 which used Alafasy).
# Format: everyayah.com/data/{Reciter}/{SSS}{AAA}.mp3

OUTDIR="$(dirname "$0")/test_corpus_v2"
RECITER="Husary_128kbps"
BASE="https://everyayah.com/data/$RECITER"

# Medium single verses (6-15 words) — different from v1
MEDIUM_VERSES=(
    "004:001"  # An-Nisa 4:1 (long bismillah + medium verse)
    "016:090"  # An-Nahl 16:90
    "031:013"  # Luqman 31:13
    "049:013"  # Al-Hujurat 49:13
    "005:003"  # Al-Ma'idah 5:3 (long verse)
    "039:053"  # Az-Zumar 39:53
    "021:087"  # Al-Anbiya 21:87
    "003:026"  # Aal-Imran 3:26
)

# Long single verses (16+ words) — different from v1
LONG_VERSES=(
    "004:012"  # An-Nisa 4:12
    "005:006"  # Al-Ma'idah 5:6
    "009:005"  # At-Tawbah 9:5
    "006:151"  # Al-An'am 6:151
    "017:023"  # Al-Isra 17:23
)

# Multi-verse sequences — different surahs from v1
MULTI_VERSES=(
    "002:001-005"  # Al-Baqarah opening
    "019:001-005"  # Maryam opening
    "056:001-004"  # Al-Waqi'ah opening
    "091:001-005"  # Ash-Shams opening
)

download_verse() {
    local surah=$1
    local ayah=$2
    local padded_surah=$(printf "%03d" $surah)
    local padded_ayah=$(printf "%03d" $ayah)
    local url="$BASE/${padded_surah}${padded_ayah}.mp3"
    local outfile="$OUTDIR/ea_${padded_surah}${padded_ayah}.mp3"

    if [ -f "$outfile" ]; then
        echo "  Already exists: $outfile"
        return 0
    fi

    echo "  Downloading $surah:$ayah -> $outfile"
    curl -sL "$url" -o "$outfile"
    if [ $? -ne 0 ] || [ ! -s "$outfile" ]; then
        echo "  FAILED: $url"
        rm -f "$outfile"
        return 1
    fi
}

download_multi() {
    local surah=$1
    local start=$2
    local end=$3
    local padded_surah=$(printf "%03d" $surah)
    local outfile="$OUTDIR/ea_multi_${padded_surah}_${start}_${end}.wav"

    if [ -f "$outfile" ]; then
        echo "  Already exists: $outfile"
        return 0
    fi

    # Download individual verses and concatenate
    local tmpfiles=""
    for ayah in $(seq $start $end); do
        download_verse $surah $ayah
        local padded_ayah=$(printf "%03d" $ayah)
        tmpfiles="$tmpfiles $OUTDIR/ea_${padded_surah}${padded_ayah}.mp3"
    done

    # Concatenate with ffmpeg
    local filter=""
    local count=0
    for f in $tmpfiles; do
        filter="$filter -i $f"
        count=$((count + 1))
    done

    echo "  Concatenating $count files -> $outfile"
    ffmpeg -hide_banner -loglevel error $filter -filter_complex "concat=n=${count}:v=0:a=1" -f wav -ar 16000 -ac 1 "$outfile" 2>/dev/null

    # Clean up individual mp3s for multi-verse (they're concatenated)
    for f in $tmpfiles; do
        rm -f "$f"
    done
}

echo "=== Downloading EveryAyah v2 samples ==="
echo "Reciter: $RECITER"
echo ""

echo "--- Medium verses ---"
for v in "${MEDIUM_VERSES[@]}"; do
    IFS=: read surah ayah <<< "$v"
    download_verse $surah $ayah
done

echo ""
echo "--- Long verses ---"
for v in "${LONG_VERSES[@]}"; do
    IFS=: read surah ayah <<< "$v"
    download_verse $surah $ayah
done

echo ""
echo "--- Multi-verse sequences ---"
for v in "${MULTI_VERSES[@]}"; do
    IFS=: read surah range <<< "$v"
    IFS=- read start end <<< "$range"
    download_multi $surah $start $end
done

echo ""
echo "Done! Files in $OUTDIR"
ls -la "$OUTDIR"/*.mp3 "$OUTDIR"/*.wav 2>/dev/null | wc -l | xargs echo "Total files:"
