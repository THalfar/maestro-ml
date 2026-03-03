#!/bin/bash
# pompottelu.sh — Opus reviewaa, Sonnet korjaa, kunnes puhdas
#
# Käyttö:
#   ./pompottelu.sh                    # kaikki src/ .py tiedostot
#   ./pompottelu.sh src/ensemble.py    # vain yksi tiedosto
#   ./pompottelu.sh src/optimization/  # kaikki kansiossa
#
# Keskeytä: Ctrl+C

MAX_ROUNDS=4
REVIEW_MODEL="opus"
CODER_MODEL="sonnet"
PROMPTS_DIR="prompts"

# --- Väripaletti ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
DIM='\033[2m'
# DISPUTE saa oman huomiovärin: kirkas magenta + bold
DISPUTE='\033[1;35m'
# BUG/LEAK = punainen, PERF = keltainen, STYLE = dim
C_BUG='\033[1;31m'
C_LEAK='\033[1;31m'
C_API='\033[1;33m'
C_PERF='\033[0;33m'
C_STYLE='\033[2;37m'
C_TODO='\033[0;36m'
NC='\033[0m'

# --- Aika-apurit ---
PIPELINE_START=$(date +%s)

timestamp() {
    echo -e "${DIM}[$(date '+%H:%M:%S')]${NC}"
}

elapsed() {
    local start=$1
    local end=$(date +%s)
    local diff=$((end - start))
    local min=$((diff / 60))
    local sec=$((diff % 60))
    if [ $min -gt 0 ]; then
        echo "${min}m ${sec}s"
    else
        echo "${sec}s"
    fi
}

# --- Värikoodattu REVIEW-raportti ---
print_colored_reviews() {
    local file=$1
    while IFS= read -r line; do
        if [[ "$line" == *"REVIEW:BUG"* ]]; then
            echo -e "  ${C_BUG}${line}${NC}"
        elif [[ "$line" == *"REVIEW:LEAK"* ]]; then
            echo -e "  ${C_LEAK}${line}${NC}"
        elif [[ "$line" == *"REVIEW:API"* ]]; then
            echo -e "  ${C_API}${line}${NC}"
        elif [[ "$line" == *"REVIEW:PERF"* ]]; then
            echo -e "  ${C_PERF}${line}${NC}"
        elif [[ "$line" == *"REVIEW:STYLE"* ]]; then
            echo -e "  ${C_STYLE}${line}${NC}"
        elif [[ "$line" == *"REVIEW:TODO"* ]]; then
            echo -e "  ${C_TODO}${line}${NC}"
        elif [[ "$line" == *"DISPUTE"* ]]; then
            echo -e "  ${DISPUTE}⚡ ${line}${NC}"
        else
            echo -e "  ${line}"
        fi
    done < <(grep -n "# REVIEW:\|# DISPUTE:" "$file" 2>/dev/null)
}

# --- Tarkista promptit ---
for PROMPT in review.md coder.md; do
    if [ ! -f "${PROMPTS_DIR}/${PROMPT}" ]; then
        echo -e "${RED}❌ Puuttuu: ${PROMPTS_DIR}/${PROMPT}${NC}"
        exit 1
    fi
done

REVIEW_PROMPT=$(cat "${PROMPTS_DIR}/review.md")
CODER_PROMPT=$(cat "${PROMPTS_DIR}/coder.md")

# --- Kerää tiedostot ---
if [ -n "$1" ]; then
    if [ -d "$1" ]; then
        FILES=$(find "$1" -name "*.py" -not -name "__init__.py" -not -path "*/test*" | sort)
    else
        FILES="$1"
    fi
else
    FILES=$(find src/ -name "*.py" -not -name "__init__.py" -not -path "*/test*" | sort)
fi

FILE_COUNT=$(echo "$FILES" | wc -l)
echo ""
echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${BOLD}🎵 MAESTRO-ML POMPOTTELU${NC}            ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                                      ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  📁 Tiedostoja: ${WHITE}${FILE_COUNT}${NC}                   ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  🔴 Review:  ${RED}${REVIEW_MODEL}${NC}                 ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  🔵 Coder:   ${BLUE}${CODER_MODEL}${NC}               ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  🔄 Max:     ${WHITE}${MAX_ROUNDS} kierrosta${NC}           ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
echo -e "$(timestamp) Pipeline käynnistyy"

APPROVED=0
NEEDS_WORK=0
DISPUTES=0
TOTAL_REVIEWS=0
TOTAL_FIXES=0
FILE_TIMES=()

FILE_IDX=0
for FILE in $FILES; do
    [ ! -s "$FILE" ] && continue

    FILE_IDX=$((FILE_IDX + 1))
    FILE_START=$(date +%s)

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  📄 [${FILE_IDX}/${FILE_COUNT}] ${BOLD}${FILE}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "$(timestamp) Aloitetaan"

    FILE_DONE=false
    for ROUND in $(seq 1 $MAX_ROUNDS); do

        # --- REVIEWER ---
        STEP_START=$(date +%s)
        echo ""
        echo -e "$(timestamp) ${RED}  🔍 REVIEWER (${REVIEW_MODEL}) — kierros ${ROUND}/${MAX_ROUNDS}${NC}"
        echo -e "${RED}  ┌──────────────────────────────┐${NC}"

        claude --model "$REVIEW_MODEL" --verbose \
            -p "${REVIEW_PROMPT}

Review this file: ${FILE}" \
            --allowedTools "Edit,Write,Read,Bash"

        TOTAL_REVIEWS=$((TOTAL_REVIEWS + 1))
        echo -e "${RED}  └──────────────────────────────┘${NC}"
        echo -e "$(timestamp) ${DIM}  Review valmis ($(elapsed $STEP_START))${NC}"

        # --- Tarkista disputet ---
        DISPUTE_COUNT=$(grep -c "# DISPUTE:" "$FILE" 2>/dev/null || echo 0)
        if [ "$DISPUTE_COUNT" -gt 0 ]; then
            DISPUTES=$((DISPUTES + DISPUTE_COUNT))
            echo ""
            echo -e "${DISPUTE}  ⚡⚡⚡ ${DISPUTE_COUNT} DISPUTE löytyi! ⚡⚡⚡${NC}"
            grep -n "# DISPUTE:" "$FILE" | while IFS= read -r line; do
                echo -e "${DISPUTE}  ${line}${NC}"
            done
        fi

        # --- Onko puhdas? ---
        if ! grep -q "# REVIEW:" "$FILE" 2>/dev/null; then
            echo ""
            echo -e "${GREEN}  ┌──────────────────────────────┐${NC}"
            echo -e "${GREEN}  │  ✅ APPROVED kierroksella ${ROUND}   │${NC}"
            echo -e "${GREEN}  │  ⏱  $(elapsed $FILE_START) kokonaisaika     │${NC}"
            echo -e "${GREEN}  └──────────────────────────────┘${NC}"
            APPROVED=$((APPROVED + 1))
            FILE_TIMES+=("$(elapsed $FILE_START)")
            FILE_DONE=true
            break
        fi

        # --- Näytä löydökset värikoodattuna ---
            DISPUTE_COUNT=$(grep -c "# DISPUTE:" "$FILE" 2>/dev/null)
            DISPUTE_COUNT=${DISPUTE_COUNT:-0}
            BUG_N=$(grep -c "REVIEW:BUG" "$FILE" 2>/dev/null)
            BUG_N=${BUG_N:-0}
            LEAK_N=$(grep -c "REVIEW:LEAK" "$FILE" 2>/dev/null)
            LEAK_N=${LEAK_N:-0}
            API_N=$(grep -c "REVIEW:API" "$FILE" 2>/dev/null)
            API_N=${API_N:-0}
            PERF_N=$(grep -c "REVIEW:PERF" "$FILE" 2>/dev/null)
            PERF_N=${PERF_N:-0}
            STYLE_N=$(grep -c "REVIEW:STYLE" "$FILE" 2>/dev/null)
            STYLE_N=${STYLE_N:-0}
            TODO_N=$(grep -c "REVIEW:TODO" "$FILE" 2>/dev/null)
            TODO_N=${TODO_N:-0}

        echo ""
        echo -e "  ${WHITE}${BOLD}${REVIEW_COUNT} kommenttia:${NC}"
        [ "$BUG_N" -gt 0 ]   && echo -e "    ${C_BUG}● BUG:   ${BUG_N}${NC}"
        [ "$LEAK_N" -gt 0 ]  && echo -e "    ${C_LEAK}● LEAK:  ${LEAK_N}${NC}"
        [ "$API_N" -gt 0 ]   && echo -e "    ${C_API}● API:   ${API_N}${NC}"
        [ "$TODO_N" -gt 0 ]  && echo -e "    ${C_TODO}● TODO:  ${TODO_N}${NC}"
        [ "$PERF_N" -gt 0 ]  && echo -e "    ${C_PERF}● PERF:  ${PERF_N}${NC}"
        [ "$STYLE_N" -gt 0 ] && echo -e "    ${C_STYLE}● STYLE: ${STYLE_N}${NC}"
        echo ""
        print_colored_reviews "$FILE"

        # --- CODER ---
        STEP_START=$(date +%s)
        echo ""
        echo -e "$(timestamp) ${BLUE}  🔧 CODER (${CODER_MODEL}) korjaa${NC}"
        echo -e "${BLUE}  ┌──────────────────────────────┐${NC}"

        claude --model "$CODER_MODEL" --verbose \
            -p "${CODER_PROMPT}

Fix all # REVIEW: comments in: ${FILE}" \
            --allowedTools "Edit,Write,Read,Bash"

        TOTAL_FIXES=$((TOTAL_FIXES + 1))
        echo -e "${BLUE}  └──────────────────────────────┘${NC}"
        echo -e "$(timestamp) ${DIM}  Korjaus valmis ($(elapsed $STEP_START))${NC}"

    done

    if [ "$FILE_DONE" = false ]; then
        echo ""
        echo -e "${RED}  ┌──────────────────────────────┐${NC}"
        echo -e "${RED}  │  ❌ EI KONVERGOITUNUT         │${NC}"
        echo -e "${RED}  │  ${MAX_ROUNDS} kierrosta käytetty       │${NC}"
        echo -e "${RED}  │  ⏱  $(elapsed $FILE_START) kokonaisaika     │${NC}"
        echo -e "${RED}  └──────────────────────────────┘${NC}"
        NEEDS_WORK=$((NEEDS_WORK + 1))
        FILE_TIMES+=("$(elapsed $FILE_START) ❌")
    fi
done

# --- Loppuyhteenveto ---
PIPELINE_ELAPSED=$(elapsed $PIPELINE_START)

echo ""
echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${BOLD}🎵 POMPOTTELU VALMIS${NC}                ${CYAN}║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════╣${NC}"
echo -e "${CYAN}║${NC}                                      ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  ${GREEN}✅ Approved:   ${APPROVED}${NC}                   ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  ${RED}❌ Needs work: ${NEEDS_WORK}${NC}                   ${CYAN}║${NC}"
if [ "$DISPUTES" -gt 0 ]; then
echo -e "${CYAN}║${NC}  ${DISPUTE}⚡ Disputes:  ${DISPUTES}${NC}                   ${CYAN}║${NC}"
fi
echo -e "${CYAN}║${NC}                                      ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  ${DIM}Reviews:  ${TOTAL_REVIEWS}${NC}                      ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  ${DIM}Fixes:    ${TOTAL_FIXES}${NC}                      ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  ${WHITE}⏱  Total: ${PIPELINE_ELAPSED}${NC}                  ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                                      ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"

# --- Per-file aikataulut ---
if [ ${#FILE_TIMES[@]} -gt 0 ]; then
    echo ""
    echo -e "${DIM}  Per-file ajat:${NC}"
    IDX=0
    for FILE in $FILES; do
        [ ! -s "$FILE" ] && continue
        if [ $IDX -lt ${#FILE_TIMES[@]} ]; then
            echo -e "${DIM}    ${FILE}: ${FILE_TIMES[$IDX]}${NC}"
        fi
        IDX=$((IDX + 1))
    done
fi

# --- Jäljellä olevat ---
REMAINING=$(grep -rl "# REVIEW:\|# DISPUTE:" src/ 2>/dev/null)
if [ -n "$REMAINING" ]; then
    echo ""
    echo -e "${RED}${BOLD}  Jäljellä:${NC}"
    grep -rn "# REVIEW:" src/ 2>/dev/null | while IFS= read -r line; do
        if [[ "$line" == *"BUG"* ]] || [[ "$line" == *"LEAK"* ]]; then
            echo -e "  ${C_BUG}${line}${NC}"
        elif [[ "$line" == *"API"* ]]; then
            echo -e "  ${C_API}${line}${NC}"
        else
            echo -e "  ${C_PERF}${line}${NC}"
        fi
    done
    grep -rn "# DISPUTE:" src/ 2>/dev/null | while IFS= read -r line; do
        echo -e "  ${DISPUTE}⚡ ${line}${NC}"
    done
fi