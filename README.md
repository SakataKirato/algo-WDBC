# Heuristic feature selection (C17)

WDBC の30特徴量に対し、Hill Climbing、Tabu Search、Simulated Annealing、Genetic Algorithm を比較します。全方式で、学習データのみを使うfoldごとの標準化、勾配降下法による無正則化ロジスティック回帰、5-fold CV平均F1から特徴量数ペナルティ（λ=0.01）を引いたfitnessを共有します。悪性 `M` を陽性クラスとします。

```sh
make
./feature_select data.csv 42,43,44
```

第2引数はカンマ区切りのシードです。各シードで層化80/20 train/test 分割を行い、探索はtrainのみで実行します。出力にはtest Accuracy / Precision / Recall / F1、選択特徴量数、実行時間、fitness評価回数とビットマスクを表示します。

全アルゴリズムのfitness評価回数は3000回に固定しています。TSとSAは評価予算に達するまで反復し、GAは個体数24の世代を評価予算内で繰り返します。HCは局所最適に達した場合、評価予算を使い切るまでランダム再スタートします。その他の探索パラメータは `feature_select.c` にまとまっています（TS: tenure 7、SA: T0=0.10・alpha=0.999、GA: 個体数24・突然変異率0.02）。
# algo-WDBC
