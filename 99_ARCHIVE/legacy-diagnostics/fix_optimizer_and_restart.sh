#!/bin/bash
# Fix optimizer state mismatch and restart training

echo "================================================================================"
echo "OPTIMIZER STATE FIX"
echo "================================================================================"
echo ""
echo "The optimizer state file is out of sync with the migrated checkpoint."
echo "This causes neuron_tangle_gate to show 0.66 instead of 0.01!"
echo ""
echo "Steps:"
echo "1. Kill current training process"
echo "2. Remove old optimizer state (backup already created)"
echo "3. Training will initialize fresh optimizer state on next run"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."

echo ""
echo "[1/3] Stopping training..."
pkill -f "python.*wakeup.py"
sleep 2

if pgrep -f "python.*wakeup.py" > /dev/null; then
    echo "⚠️  Process still running, forcing kill..."
    pkill -9 -f "python.*wakeup.py"
    sleep 1
fi

if pgrep -f "python.*wakeup.py" > /dev/null; then
    echo "❌ Could not stop training! Please kill manually."
    exit 1
else
    echo "✓ Training stopped"
fi

echo ""
echo "[2/3] Removing old optimizer state..."
if [ -f "SHKAIRA/soul/babyllm_S_4200.pth.optim" ]; then
    echo "Backup already created at: babyllm_S_4200.pth.optim.OLD_BEFORE_FIX"
    rm "SHKAIRA/soul/babyllm_S_4200.pth.optim"
    echo "✓ Old optimizer state removed"
else
    echo "✓ Optimizer state already removed"
fi

echo ""
echo "[3/3] Ready to restart!"
echo ""
echo "================================================================================"
echo "NEXT STEPS"
echo "================================================================================"
echo ""
echo "Run this command to restart training:"
echo "  python3 wakeup.py"
echo ""
echo "What will happen:"
echo "  - Loads checkpoint with neuron_tangle_gate = -4.58 (sigmoid 0.01) ✓"
echo "  - Initializes FRESH optimizer state (no old momentum)"
echo "  - TANGLE_neuron_gate should show ~0.01 in stats ✓"
echo "  - Training should be MUCH faster (seconds, not minutes per step)"
echo ""
echo "================================================================================"
