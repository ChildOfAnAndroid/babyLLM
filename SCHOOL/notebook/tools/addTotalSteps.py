import re

with open('/Users/charis/Documents/GitHub/Shkaira/SCHOOL/statistics/LOGS/training/trainingLog_100.txt', 'r') as f:
    lines = f.readlines()

output = []
running_total = 0
previous_step = 0

for line in lines:
    if 'Total Steps:' in line:
        output.append(line)
        continue

    # Match line with timestamp
    timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(\d+)\s*\|', line)
    step_match_old = re.search(r'\bStep (\d+)', line)

    if timestamp_match:
        # New format: | <step> | with no label
        timestamp, current_step = timestamp_match.groups()
        current_step = int(current_step)

    elif step_match_old:
        # Old format: Step xxx
        current_step = int(step_match_old.group(1))

    else:
        output.append(line)
        continue

    # Compute delta
    step_delta = current_step - previous_step
    if step_delta < 0:
        step_delta = current_step
    running_total += step_delta
    previous_step = current_step

    # Inject total steps after timestamp (old or new)
    line = re.sub(
        r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
        rf'\1 | Total Steps: {running_total}',
        line
    )

    output.append(line)

with open('/Users/charis/Documents/GitHub/Shkaira/SCHOOL/statistics/LOGS/training/trainingLog_100_withTotalSteps.txt', 'w') as f:
    f.writelines(output)

print("✅ Global running total added ONLY where missing. No resets.")