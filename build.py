import pandas as pd

data = {
    "question": ["霄云里8号的建筑高度是49.5米?", "梅赛德斯-奔驰如何接管布朗GP车队？"],
    "answer": [
        "霄云里8号的建筑高度是49.5米。",
        "梅赛德斯-奔驰通过收购布朗GP 75.1%的股份接管了车队。",
    ],
}

df = pd.DataFrame(data)

df.to_csv("example.csv", index=False)
