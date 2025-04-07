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
