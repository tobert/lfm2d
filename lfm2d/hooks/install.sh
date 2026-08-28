#!/usr/bin/env bash
# Wire the lfm2d advisory PreToolUse hook into ~/.claude/settings.json.
#
#   install.sh             # show what is installed now
#   install.sh bootstrap   # FRESH machine: add the hook entry (no prior hook)
#   install.sh install     # SWAP the dotfiles regex hook for the advisory one
#   install.sh uninstall   # undo `install` (restore the swapped-out command)
#
# Two ways in, deliberately separate:
#
# `install` is a parity-gated SWAP. It replaces the dotfiles regex hook
# with the advisory one, which makes the SAME decisions -- gated by
# test_parity.py, 33 cases byte-for-byte -- and additionally asks lfm2d
# for a second opinion that it only writes to a log. It refuses when
# there is no dotfiles baseline to gate against or no existing hook entry
# to swap.
#
# `bootstrap` is for a machine that has never had a hook: it ADDS the
# entry (creating settings.json if needed) and refuses if any recognized
# hook is already wired -- that machine wants `install`. There is no
# parity gate because there is no baseline; the regex rules are embedded
# verbatim in the hook, so the decisions are the dotfiles decisions either
# way. The daemon endpoint is written INTO the command string from
# LFM2D_URL (default loopback), because the hook's own default is loopback
# and a remote daemon is this machine's configuration, not the code's.
#
# Rollback for either is one string in settings.json; a timestamped backup
# is taken before any edit. Tested in test_hook_config.py against a temp HOME.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# python3, not `python`: both are 3.14.6 here, but the tests were run under
# python3 and a guard should not inherit whatever `python` resolves to later.
ADVISORY="$REPO/lfm2d/hooks/pre_command_advisory.py"
ADVISORY_CMD="python3 $ADVISORY"
# The settings entry points at ~/.claude/hooks/pre-command.py, which is a
# symlink into dotfiles (verified identical). The parity gate runs against
# the dotfiles path; the settings string is what we restore.
BASELINE="$HOME/src/tobert/dotfiles/.claude/hooks/pre-command.py"
SETTINGS="$HOME/.claude/settings.json"
# The exact command string that was installed before we touched it, so
# `uninstall` restores what was there rather than what this script guesses
# was there.
PREVIOUS="$HOME/.claude/.lfm2d-hook-previous"
# Hook timeout in seconds -- the hook's own per-call budget is capped at 8 s
# for the longest commands (LFM2D_TIMEOUT_CAP) but advisory mode fails open,
# so 5 s here bounds what a wedged daemon can cost a Bash call.
HOOK_TIMEOUT=5

die() { echo "error: $*" >&2; exit 1; }

current_command() {
  [ -f "$SETTINGS" ] || { echo ''; return; }
  python3 - "$SETTINGS" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception:
    print(''); raise SystemExit
for group in s.get('hooks', {}).get('PreToolUse', []):
    for h in group.get('hooks', []):
        cmd = h.get('command', '')
        if 'pre-command.py' in cmd or 'pre_command_advisory.py' in cmd:
            print(cmd); raise SystemExit
print('')
PY
}

set_command() {
  python3 - "$SETTINGS" "$1" <<'PY'
import json, sys
path, new = sys.argv[1], sys.argv[2]
s = json.load(open(path))
changed = 0
for group in s.get('hooks', {}).get('PreToolUse', []):
    for h in group.get('hooks', []):
        cmd = h.get('command', '')
        if 'pre-command.py' in cmd or 'pre_command_advisory.py' in cmd:
            h['command'] = new
            changed += 1
if changed != 1:
    # Refuse rather than guess. Zero means the hook is wired somewhere this
    # script does not understand; two or more means editing one is as likely
    # to be wrong as right, and a half-swapped guard is worse than either
    # state.
    raise SystemExit(f'refusing: expected exactly 1 matching hook entry, found {changed}')
json.dump(s, open(path, 'w'), indent=2)
open(path, 'a').write('\n')
print(f'set -> {new}')
PY
}

# Add a PreToolUse/Bash group carrying the advisory hook to settings.json,
# creating the file if absent. Every other key is preserved verbatim.
add_entry() {
  python3 - "$SETTINGS" "$1" "$HOOK_TIMEOUT" <<'PY'
import json, os, sys
path, cmd, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])
if os.path.exists(path):
    with open(path) as f:
        s = json.load(f)
else:
    s = {}
hooks = s.setdefault('hooks', {})
pre = hooks.setdefault('PreToolUse', [])
pre.append({'matcher': 'Bash', 'hooks': [{'type': 'command', 'command': cmd, 'timeout': timeout}]})
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as f:
    json.dump(s, f, indent=2)
    f.write('\n')
print(f'added -> {cmd}')
PY
}

settings_valid_json() {
  python3 -c "import json,sys;json.load(open(sys.argv[1]))" "$SETTINGS" 2>/dev/null
}

backup_settings() {
  cp -p "$SETTINGS" "$SETTINGS.bak-$(date +%Y%m%d-%H%M%S)"
}

# What every subcommand needs: the hook itself, and a settings file that is
# JSON if it exists. Whether a settings file or a dotfiles baseline must
# ALSO exist depends on the subcommand, so those checks live there.
preflight_common() {
  [ -f "$ADVISORY" ] || die "advisory hook missing at $ADVISORY"
  if [ -f "$SETTINGS" ] && ! settings_valid_json; then
    die "$SETTINGS is not valid JSON -- fix that before touching a guard"
  fi
}

case "${1:-status}" in
  status)
    preflight_common
    cur="$(current_command)"
    echo "settings: $SETTINGS$([ -f "$SETTINGS" ] || echo '  (absent)')"
    echo "hook:     ${cur:-<none found>}"
    case "$cur" in
      *pre_command_advisory.py*) echo "state:    ADVISORY (lfm2d second opinion, logged not enforced)" ;;
      *pre-command.py*)          echo "state:    BASELINE (regex only)" ;;
      '')                        echo "state:    NONE -- 'bootstrap' for a fresh machine, 'install' to swap a dotfiles hook" ;;
      *)                         echo "state:    unrecognized" ;;
    esac
    echo
    echo "advisory log: ${XDG_CACHE_HOME:-$HOME/.cache}/claude-hooks/lfm2d-advisory.jsonl"
    ;;

  bootstrap)
    preflight_common
    cur="$(current_command)"
    case "$cur" in
      *pre_command_advisory.py*) die "already wired: $cur -- nothing to do" ;;
      '') ;;
      *)  die "a hook is already wired ($cur); use 'install' to swap it, not bootstrap" ;;
    esac
    url="${LFM2D_URL:-http://127.0.0.1:8088}"
    cmd="LFM2D_URL=$url $ADVISORY_CMD"
    if [ -f "$SETTINGS" ]; then
      backup_settings
    fi
    add_entry "$cmd"
    echo
    echo "Bootstrapped. lfm2d is advisory only; the regex rules decide."
    echo "Daemon:   $url  (change by editing the command string in $SETTINGS)"
    [ -f "$SETTINGS.bak-"* ] 2>/dev/null && echo "Backup:   $SETTINGS.bak-*"
    echo "Verify:   LFM2D_URL=$url python3 $REPO/lfm2d/hooks/test_advisory_live.py"
    ;;

  install)
    [ -f "$BASELINE" ] || die "dotfiles hook missing at $BASELINE -- 'install' is a parity-gated swap; a machine with no baseline wants 'bootstrap'"
    [ -f "$SETTINGS" ] || die "no settings at $SETTINGS -- 'install' swaps an existing hook; a fresh machine wants 'bootstrap'"
    preflight_common
    # Parity is the whole basis for calling this swap safe, so it is a GATE,
    # not a suggestion. If the two hooks ever decide differently, this script
    # must not be the thing that finds out in production.
    echo "== parity gate =="
    python3 "$REPO/lfm2d/hooks/test_parity.py" "$BASELINE" >/dev/null \
      || die "parity gate FAILED -- not installing. Run it directly to see the diff."
    echo "33/33 identical decisions vs $BASELINE"
    echo
    cur="$(current_command)"
    case "$cur" in
      *pre_command_advisory.py*) echo "already installed; nothing to do."; exit 0 ;;
    esac
    [ -n "$cur" ] || die "no recognizable PreToolUse hook in settings -- 'install' swaps one; a fresh machine wants 'bootstrap'"

    backup_settings
    printf '%s\n' "$cur" > "$PREVIOUS"
    set_command "$ADVISORY_CMD"
    echo
    echo "Installed. Decisions are unchanged; lfm2d is advisory only."
    echo "Previous: $cur  (saved to $PREVIOUS)"
    echo "Backup:   $SETTINGS.bak-*"
    echo "Rollback: $0 uninstall"
    ;;

  uninstall)
    [ -f "$SETTINGS" ] || die "no settings at $SETTINGS"
    preflight_common
    if [ -f "$PREVIOUS" ]; then
      restore="$(cat "$PREVIOUS")"
    else
      # Nothing recorded -- fall back to the settings string that was in use
      # before this script existed, rather than an absolute dotfiles path
      # that would silently change how the hook is invoked.
      restore="python ~/.claude/hooks/pre-command.py"
      echo "note: no saved previous command; restoring the known default"
    fi
    backup_settings
    set_command "$restore"
    rm -f "$PREVIOUS"
    echo "Reverted to: $restore"
    ;;

  *) die "usage: $0 [status|bootstrap|install|uninstall]" ;;
esac
