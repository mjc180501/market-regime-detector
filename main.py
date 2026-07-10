import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

spy = yf.download('SPY', start='2005-01-01',multi_level_index=False)
spy.to_csv('SPY.csv')

# lets build market descriptors
# df = pd.read_csv('SPY.csv', index_col='Date', parse_dates=True)
df = pd.read_csv(
    "SPY.csv",
    parse_dates=["Date"]
)

df.set_index("Date", inplace=True)
print(df.head())
print(df.columns)

df["Returns"] = df["Close"].pct_change()
df["Volatility"] = df["Returns"].rolling(20).std()
df["Momentum"] = df["Close"]/df["Close"].shift(20)-1
df["MA20"] = df["Close"].rolling(20).mean()
df["MA100"] = df["Close"].rolling(100).mean()
df["Trend"] = df["MA20"] / df["MA100"]
df["VolumeChange"] = df["Volume"].pct_change()


df_clean = df.dropna().copy()
# let's standardize 
features = ["Returns", "Volatility", "Momentum", "Trend", "VolumeChange"]
scaler = StandardScaler()
X = df_clean[features]
X_scaled = scaler.fit_transform(X)

# see below for how i got n=4
model = KMeans(n_clusters=4, random_state=69)

df_clean["Cluster"] = model.fit_predict(X_scaled)

# lets try to see what the clusters actually mean
profile = df_clean.groupby("Cluster")[features].mean()
print(profile)


# plot, only run if you wanna see the plot
def plot():
    plt.figure(figsize=(12, 6))
    plt.scatter(
        df_clean.index,
        df_clean["Close"],
        c=df_clean["Cluster"],
        cmap="viridis",
        s=10
    )
    plt.title("Market Regime Clustering")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.colorbar(label="Cluster")
    plt.show()

def find_k():
    # lets try to see if we can get the optimal number of clusters
    # lets plot inertia against k to see
    # im going to try 2-100 clusters

    k = []
    inertias = []
    for i in range(2, 101):
        k.append(i)
        model = KMeans(n_clusters=i, random_state=69)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(12,6))
    plt.scatter(k, inertias)
    plt.title("Inertias vs K")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.show()


# the inertia plot shows that 4-8 is probably a good place to keep the k values

# the cluster kmeans will seperate along the two axes that drive most of the variance in market data: direction and turbulence
# some archetypes we see could be
# calm uptrend (positive returns, low volatility) - grinding bull market
# crash / high-vol selloff (negative returns, volatility spike, momentum negative) 
# choppy / range-bound (returns ~= 0, volatility moderate, trend ration ~= 1)
# volatile recovery / rebound (positive returns but volatility is elevated)

# at this point the plot looks good, it describes trends/clusters that we should expect with reasonale seperation from one another
# however there is one problem. 
# when making these clusters, we allow kmeans to see ALL of the available data, even including future data
# in a more realistic scenario, we would want to only let the model see past behavior and predict based on data it hasn't seen yet
# lets create a slidng window in which for a certain number of trading days we refit the clusters so that more recent data has more of an impact on prediction than past data

def walk_forward():
    df_roll = df_clean.copy()
    df_roll["Cluster"] = None

    rolling_duration = 250
    refit_freq = 20
    min_window = 250

    counter = 0
    first = True

    for (i, (index,row)) in enumerate(df_roll.iterrows()):
        if i >= min_window:
            counter += 1
            subset = df_roll.iloc[i - rolling_duration : i]
            # fit kmeans, standard scalar
            if counter > refit_freq or first == True:
                first = False
                counter = 0
                X = subset[features]
                X_s = scaler.fit_transform(X)
                model = KMeans(n_clusters=4, random_state=69)
                model.fit(X_s)
            x = df_roll.iloc[[i]][features]
            x_s = scaler.transform(x)
            cluster = model.predict(x_s)[0]
            df_roll.loc[df_roll.index[i], "Cluster"] = cluster

    print(df_roll["Cluster"])
    print(df_roll["Cluster"].value_counts())
    return df_roll

# i put it in a function just to keep things more polished
df_roll = walk_forward()

# now that ive been able to make the clusters this way, i want to compare these clusters with the ones i created from the full data
# to do this comparison id like to do ari/nmi 

c1 = df_roll["Cluster"].dropna()
c2 = df_clean.loc[c1.index, "Cluster"]

nmi = normalized_mutual_info_score(c1, c2)
ari = adjusted_rand_score(c1, c2)
print(f"NMI: {nmi}")
print(f"ARI: {ari}")

# lets also do a timeline strip visual
fig, axes = plt.subplots(2, 1, figsize=(15,8), sharex=True)

axes[0].scatter(
    df_clean.loc[c1.index].index,
    df_clean.loc[c1.index, "Close"],
    c=c2.astype(int),
    cmap="viridis",
    s=8
)
axes[0].plot(df_clean.loc[c1.index].index,
             df_clean.loc[c1.index, "Close"],
             color="gray",
             alpha=0.4)
axes[0].set_title("Full-data clustering")

axes[1].scatter(
    df_roll.loc[c1.index].index,
    df_roll.loc[c1.index, "Close"],
    c=c1.astype(int),
    cmap="viridis",
    s=8
)
axes[1].plot(df_roll.loc[c1.index].index,
             df_roll.loc[c1.index, "Close"],
             color="gray",
             alpha=0.4)
axes[1].set_title("Walk-forward clustering")

plt.tight_layout()
plt.show()