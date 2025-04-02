printLossFreq = 10

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
            "Loss": random.random()
        })
    avgStats = {}
    for key, value in stats.items():
        avgStats[key] = value / printLossFreq if printLossFreq > 0 else 0

    print(" ".join([f"{key}: {value:.2f}" for key, value in avgStats.items()]))

    #stats.clear()