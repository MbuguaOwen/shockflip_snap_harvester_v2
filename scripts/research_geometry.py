import pandas as pd

geom = pd.read_csv("results/meta/snap_meta_events_geometry.csv")

print(geom.shape)
print(geom.columns)
print(geom['risk_profile'].value_counts())
print(geom['barrier_y_H30_R3p0_sl0p5'].value_counts())
