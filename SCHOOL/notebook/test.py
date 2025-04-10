trainingLogFreq_1000 = 10

from collections import Counter
import random
stats = Counter({
    "Loss": 0,
    "Grad Norm": 0,
    "Logit Min": 0,
    "Logit Max": 0,
})

for _ in range(10):
    for _ in range(10):
        stats.update({
            "Loss": random.random(),
            "Grad Norm": random.random(),
        })
    avgStats = {}
    for key, value in stats.items():
        avgStats[key] = value / trainingLogFreq_1000 if trainingLogFreq_1000 > 0 else 0

    print(" ".join([f"{key}: {(value/trainingLogFreq_1000 if trainingLogFreq_1000 > 0 else 0):.2f}" for key, value in stats.most_common()]))

    stats.clear()

"""    import code
    vars = globals().copy()
    vars.update(locals())
    code.interact(local=vars)"""

"""for attr in dir(babyLLM):
    if not attr.startswith("__"):
        print(attr, "→", getattr(babyLLM, attr))"""

"""geepy thingy test"""

"""def deep_inspect(obj, prefix="babyLLM", depth=0, max_depth=2):
        indent = "    " * depth
        if depth > max_depth:
            return

        print(f"{indent}inspecting... {prefix} ({type(obj).__name__})")

        for attr in dir(obj):
            if attr.startswith("__"):
                continue

            try:
                value = getattr(obj, attr)
                if callable(value):
                    print(f"{indent}  {attr} → <function>")
                elif isinstance(value, (int, float, str, bool, torch.Tensor)):
                    val_str = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                    print(f"{indent}  {attr} → {val_str}")
                else:
                    print(f"{indent}  {attr} → {type(value).__name__}")
                    babyLLM.deep_inspect(value, prefix=f"{prefix}.{attr}", depth=depth+1, max_depth=max_depth)
            except Exception as e:
                print(f"{indent}  {attr} → <error reading value: {e}>")"""

"""def HUD_fixScroll(self):
    height = shutil.get_terminal_size().lines
    #reserved_hud_lines = 5
    #training_lines_height = height - reserved_hud_lines

    #sys.stdout.write("\033[?25l\033[H\033[2J")  # Hide cursor, clear, move to top
    #sys.stdout.flush()

    # You should print training lines here *before* calling this if you want control

    # Move to bottom section and draw HUD
    training_lines_height = training_lines_height
    sys.stdout.write(f"\033[{training_lines_height + 1};0H")  # Move to HUD zone
    sys.stdout.flush()

    self.printHUD(
        windowWeights=(F.softmax(self.parallelNeuronLayer.windowWeighting, dim=0) + 0.1).detach().cpu().numpy(),
        guessHUD=self.guessHUD
    )

    sys.stdout.write(f"\033[{height};0H")  # Move cursor just above HUD for next cycle
    sys.stdout.flush()"""
