#!/usr/bin/env bash
# Compare generated SASS of if-constexpr vs plain-if kernels. Usage: ./check_sass.sh build/bench_branch
# Patterns are matched against DEMANGLED kernel names (c++filt).
BIN=${1:-build/bench_branch}
cuobjdump -sass "$BIN" | c++filt > /tmp/all_demangled.sass

extract() {  # $1 = fixed substring of the demangled kernel name
  awk -v p="$1" '/Function :/{on=(index($0,p)>0)} on' /tmp/all_demangled.sass \
    | sed -E 's#/\*[^*]*\*/##g' | grep -v 'Function :'
}
cmp_pair() {  # $1 = label, $2 = Cexpr pattern, $3 = If pattern
  local a b na nb
  a=$(extract "$2"); b=$(extract "$3")
  na=$(printf '%s\n' "$a" | grep -c .); nb=$(printf '%s\n' "$b" | grep -c .)
  if [ "$na" -eq 0 ] || [ "$nb" -eq 0 ]; then echo "ERROR  $1: pattern matched nothing ($na/$nb lines)"; return; fi
  if [ "$a" == "$b" ]; then echo "SAME   $1 ($na lines)"; else echo "DIFFER $1 ($na vs $nb lines)"; fi
}
calls() { extract "$1" | grep -c 'CALL'; }   # double sin is a CALL to a libdevice routine

cmp_pair "Bool true"   "Cexpr_Bool<true>"   "If_Bool<true>"
cmp_pair "Bool false"  "Cexpr_Bool<false>"  "If_Bool<false>"
cmp_pair "Mode 0"      "Cexpr_Mode<0>"      "If_Mode<0>"
cmp_pair "Mode 1"      "Cexpr_Mode<1>"      "If_Mode<1>"
cmp_pair "Mode 2"      "Cexpr_Mode<2>"      "If_Mode<2>"
cmp_pair "Type float"  "Cexpr_Type, float>"  "If_Type, float>"
cmp_pair "Type double" "Cexpr_Type, double>" "If_Type, double>"

echo "--- controls: CALL count (heavy double sin => >0, light => 0)"
for p in "If_Bool<true>" "If_Bool<false>" "If_Type, float>" "If_Type, double>"; do
  echo "  $p : $(calls "$p") CALL"
done
echo "--- registers / stack"
cuobjdump --dump-resource-usage "$BIN" | c++filt | grep -E "Function|REG" | grep -E "Cexpr_|If_" -A1 | grep -v '^--'
