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

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  MAESTRO-ML POMPOTTELU${NC}"
echo -e "${BLUE}  Tiedostoja: ${FILE_COUNT}${NC}"
echo -e "${BLUE}  Review: ${REVIEW_MODEL} | Coder: ${CODER_MODEL}${NC}"
echo -e "${BLUE}======================================${NC}"

APPROVED=0
NEEDS_WORK=0

for FILE in $FILES; do
    [ ! -s "$FILE" ] && continue

    echo -e "\n${YELLOW}━━━ ${FILE} ━━━${NC}"

    FILE_DONE=false
    for ROUND in $(seq 1 $MAX_ROUNDS); do

        # --- REVIEWER ---
        echo -e "\n${RED}  ▸ REVIEWER (${REVIEW_MODEL}) kierros ${ROUND}${NC}"
        echo -e "${RED}  ──────────────────────────────${NC}"
        claude --model "$REVIEW_MODEL" --verbose -p "$(cat ${PROMPTS_DIR}/review.md)

BEFORE reviewing, build context:
1. Read CLAUDE.md for project principles
2. Read the file: ${FILE}
3. Read every import — open each imported module
   and understand the interfaces this file depends on
4. Read configs/schemas/ for YAML contracts if the
   file uses config parameters
5. NOW review with full understanding

Review this file: ${FILE}" --allowedTools "Edit,Write,Read,Bash"

        # --- Onko puhdas? ---
        if ! grep -q "# REVIEW:" "$FILE" 2>/dev/null; then
            echo -e "\n${GREEN}  ✓ APPROVED kierroksella ${ROUND}${NC}"
            APPROVED=$((APPROVED + 1))
            FILE_DONE=true
            break
        fi

        echo -e "\n${YELLOW}  ⚠ $(grep -c '# REVIEW:' "$FILE") kommenttia:${NC}"
        grep -n "# REVIEW:" "$FILE"

        # --- CODER ---
        echo -e "\n${BLUE}  ▸ CODER (${CODER_MODEL}) korjaa${NC}"
        echo -e "${BLUE}  ──────────────────────────────${NC}"
        claude --model "$CODER_MODEL" --verbose -p "$(cat ${PROMPTS_DIR}/coder.md)

BEFORE fixing, build context:
1. Read CLAUDE.md for project principles
2. Read ${FILE} and understand all its imports
3. Read configs/ if file uses config parameters

MODE 2: Fix all # REVIEW: comments in ${FILE}
Fix in priority: BUG/LEAK → TODO → PERF.
Remove each REVIEW comment after fixing." --allowedTools "Edit,Write,Read,Bash"

   
    done

    if [ "$FILE_DONE" = false ]; then
        echo -e "\n${RED}  ✗ Ei konvergoitunut ${MAX_ROUNDS} kierroksessa${NC}"
        NEEDS_WORK=$((NEEDS_WORK + 1))
    fi
done

# --- Yhteenveto ---
echo -e "\n${BLUE}======================================${NC}"
echo -e "${GREEN}  ✓ Approved:   ${APPROVED}${NC}"
echo -e "${RED}  ✗ Needs work: ${NEEDS_WORK}${NC}"
echo -e "${BLUE}======================================${NC}"

REMAINING=$(grep -rl "# REVIEW:" src/ 2>/dev/null)
if [ -n "$REMAINING" ]; then
    echo -e "\n${RED}  Jäljellä:${NC}"
    grep -rn "# REVIEW:" src/
fi