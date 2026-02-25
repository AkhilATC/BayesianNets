"""
1. 🌳 Structure is random

2. 🔢 Node values depend on parent

3. 🧠 Small enough to understand fully


Hidden Root
   |
 (A or B chosen dynamically)
   |
  Leaf (Observed data)



"""
import numpy as np

def sample_dynamic_tree():
    # 1. Sample structure
    Z = np.random.choice(["A", "B"], p=[0.5, 0.5])
    # 2. Sample root
    x_root = np.random.normal(0, 1)

    # 3. Sample intermediate node
    x_intermediate = np.random.normal(x_root, 1)

    # 4. Sample leaf observation
    y = np.random.normal(x_intermediate, 1)

    return {
        "structure": Z,
        "root": x_root,
        "intermediate": x_intermediate,
        "leaf": y
    }

for i in range(5):
    sample = sample_dynamic_tree()
    print(sample)

