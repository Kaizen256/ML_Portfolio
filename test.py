import random
import textwrap


# -----------------------------
# Flashcard Quiz: Model Category
# -----------------------------


CATEGORIES = {
    # ---------- Unsupervised ----------
    "clustering": [
        "KMeans",
        "MiniBatchKMeans",
        "AgglomerativeClustering",
        "DBSCAN",
        "OPTICS",
        "MeanShift",
        "SpectralClustering",
        "GaussianMixture (GMM)",
    ],
    "dimensionality_reduction": [
        "PCA",
        "KernelPCA",
        "SparsePCA",
        "IncrementalPCA",
        "TruncatedSVD",
        "FactorAnalysis",
        "FastICA (ICA)",
        "NMF",
        "t-SNE",
        "UMAP",
    ],
    "anomaly_detection": [
        "IsolationForest",
        "OneClassSVM",
        "LocalOutlierFactor (LOF)",
        "EllipticEnvelope",
    ],


    # ---------- Supervised ----------
    # Put "pure" classifiers here (primarily classification)
    "clf": [
        "LogisticRegression",
        "LinearSVC",
        "SVC",
        "KNeighborsClassifier",
        "GaussianNB",
        "MultinomialNB",
        "BernoulliNB",
        "DecisionTreeClassifier",
        "MLPClassifier",
        "LinearDiscriminantAnalysis (LDA)",
        "QuadraticDiscriminantAnalysis (QDA)",
    ],


    # Put "pure" regressors here (primarily regression)
    "reg": [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SVR",
        "KNeighborsRegressor",
        "DecisionTreeRegressor",
        "MLPRegressor",
        "BayesianRidge",
        "HuberRegressor",
        "RANSACRegressor",
        "TheilSenRegressor",
        "PoissonRegressor",
        "GammaRegressor",
        "TweedieRegressor",
    ],


    # Ensembles can do classification or regression, so keep them in one bucket
    "ensemble": [
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "AdaBoostClassifier",
        "AdaBoostRegressor",
        "BaggingClassifier",
        "BaggingRegressor",
        "StackingClassifier",
        "StackingRegressor",
        "VotingClassifier",
        "VotingRegressor",
    ],
}


# Accepted aliases for answers (add your own if you want)
ALIASES = {
    "clustering": {"clustering", "cluster", "c"},
    "dimensionality_reduction": {"dimensionality_reduction", "dim reduction", "dimensionality", "dr", "d"},
    "anomaly_detection": {"anomaly_detection", "anomaly", "outlier", "ad", "a"},


    "clf": {"clf", "class", "classification", "classifier"},
    "reg": {"reg", "regr", "regression", "regressor"},
    "ensemble": {"ensemble", "ens", "forest", "boosting", "bagging", "stacking", "voting"},
}


def normalize(s: str) -> str:
    return " ".join(s.strip().lower().replace("_", " ").split())


def resolve_category(user_answer: str):
    ans = normalize(user_answer)
    for cat, names in ALIASES.items():
        if ans in {normalize(x) for x in names}:
            return cat
    return None


def build_deck():
    deck = []
    for cat, models in CATEGORIES.items():
        for m in models:
            deck.append((m, cat))
    random.shuffle(deck)
    return deck


def pretty_cat(cat: str) -> str:
    return cat.replace("_", " ")


def main():
    deck = build_deck()
    total = 0
    correct = 0


    help_text = """
    Categories (type one):
      - clustering (c)
      - dimensionality_reduction (dr or d)
      - anomaly_detection (a)
      - clf (classification)
      - reg (regression)
      - ensemble (ens)


    Commands:
      - 'h' / 'help' to show this
      - 'q' to quit
      - 's' / 'skip' to skip a card
    """.strip()


    print("\nModel Category Flashcards\n" + "-" * 28)
    print(textwrap.dedent(help_text))
    print()


    i = 0
    while i < len(deck):
        model, true_cat = deck[i]
        print(f"\n[{i+1}/{len(deck)}]  Model: {model}")
        ans = input("Category (c / dr / a / clf / reg / ens): ").strip()


        if not ans:
            print("Blank answer. Try again.")
            continue


        low = normalize(ans)
        if low in {"q", "quit", "exit"}:
            break
        if low in {"h", "help", "?"}:
            print("\n" + textwrap.dedent(help_text))
            continue
        if low in {"skip", "s"}:
            print(f"Skipped. Answer: {pretty_cat(true_cat)}")
            total += 1
            i += 1
            continue


        guessed = resolve_category(ans)
        total += 1


        if guessed == true_cat:
            correct += 1
            print("✅ Correct.")
        else:
            guessed_display = pretty_cat(guessed) if guessed else f"'{ans}' (unrecognized)"
            print(f"❌ Nope. You said {guessed_display}. Correct: {pretty_cat(true_cat)}")


        print(f"Score: {correct}/{total} ({(correct/total*100):.1f}%)")
        i += 1


    print("\nDone.")
    if total > 0:
        print(f"Final: {correct}/{total} ({(correct/total*100):.1f}%)")
    else:
        print("Final: 0 questions answered. Truly iconic.")


if __name__ == "__main__":
    main()
