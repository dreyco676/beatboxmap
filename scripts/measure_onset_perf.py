from voxkit.eval.harness import run_for_tier
result = run_for_tier("minimum-reproducible")
print("F-measure:", round(result["f_measure"], 4))
print("MAE ms:", round(result["mae_ms"], 2))
