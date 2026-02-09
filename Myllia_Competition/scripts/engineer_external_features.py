import os, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external"
OUT = ROOT / "features" / "external"
OUT.mkdir(parents=True, exist_ok=True)

MEANS_PATH  = ROOT / "data" / "training_data_means.csv"
VALMAP_PATH = ROOT / "data" / "pert_ids_val.csv"

def load_union_genes():
    df = pd.read_csv(MEANS_PATH)
    gene_cols = [c for c in df.columns if c != "pert_symbol"]
    df_val = pd.read_csv(VALMAP_PATH)
    val_genes = df_val["pert"].astype(str).tolist()
    train_genes = df.loc[df["pert_symbol"].astype(str) != "non-targeting", "pert_symbol"].astype(str).tolist()
    union = sorted(set(gene_cols) | set(val_genes) | set(train_genes))
    return union, gene_cols, train_genes

def build_hgnc_maps(hgnc_path: Path):
    df = pd.read_csv(hgnc_path, sep="\t", dtype=str).fillna("")
    # symbol + alias_symbol + prev_symbol + entrez_id + ensembl_gene_id + uniprot_ids
    sym = df["symbol"].str.upper().tolist()

    alias_cols = ["alias_symbol", "prev_symbol"]
    map_to_symbol = {}
    uniprot_to_symbol = {}

    for i, s in enumerate(sym):
        map_to_symbol[s] = s
        # aliases
        for col in alias_cols:
            aliases = [a.strip().upper() for a in str(df.loc[i, col]).split("|") if a.strip()]
            for a in aliases:
                map_to_symbol.setdefault(a, s)
        # uniprot ids
        ups = [u.strip() for u in str(df.loc[i, "uniprot_ids"]).split("|") if u.strip()]
        for u in ups:
            uniprot_to_symbol.setdefault(u, s)

    return map_to_symbol, uniprot_to_symbol

def genept_pca(pkl_path: Path, union: list[str], alias_map: dict, pca_dim=128, seed=6):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        obj = dict(obj)

    keys = []
    vecs = []
    for k, v in obj.items():
        if v is None: 
            continue
        keys.append(str(k).upper())
        vecs.append(np.asarray(v, dtype=np.float32).ravel())
    M = np.stack(vecs, axis=0)

    pca = PCA(n_components=pca_dim, random_state=seed)
    pca.fit(M)
    mean_raw = M.mean(axis=0)

    E = np.zeros((len(union), pca_dim), dtype=np.float32)
    hit = 0
    for i, g in enumerate([u.upper() for u in union]):
        g2 = alias_map.get(g, g)
        raw = obj.get(g2, None)
        if raw is None:
            raw = mean_raw
        else:
            hit += 1
            raw = np.asarray(raw, dtype=np.float32).ravel()
        E[i] = pca.transform(raw[None, :])[0].astype(np.float32)
    return E, hit

def string_graph_features(aliases_path_txt: Path, links_path_txt: Path, union: list[str], alias_map: dict):
    alias_df = pd.read_csv(aliases_path_txt, sep="\t", header=None, names=["pid", "alias", "src"], dtype=str)
    alias_df["alias"] = alias_df["alias"].fillna("").astype(str).str.upper()
    alias2pid = {}
    for pid, al in zip(alias_df["pid"], alias_df["alias"]):
        if not al:
            continue
        if al.isalnum() and len(al) <= 20:
            alias2pid.setdefault(al, pid)

    gene2pid = {}
    for g in [u.upper() for u in union]:
        g2 = alias_map.get(g, g)
        pid = alias2pid.get(g2)
        if pid:
            gene2pid[g] = pid

    link_df = pd.read_csv(
        links_path_txt,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["p1", "p2", "score"],
        skiprows=1,
        dtype={"p1": str, "p2": str, "score": np.int32},
    )

    link_df["score"] = link_df["score"].astype(np.int32)

    G = nx.Graph()
    for p1, p2, sc in zip(link_df["p1"], link_df["p2"], link_df["score"]):
        if sc < 700:
            continue
        G.add_edge(p1, p2, weight=float(sc))

    deg = dict(G.degree())
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    andeg = nx.average_neighbor_degree(G)

    F = np.zeros((len(union), 3), dtype=np.float32)
    hit = 0
    for i, g in enumerate([u.upper() for u in union]):
        pid = gene2pid.get(g)
        if pid and pid in G:
            hit += 1
            F[i, 0] = float(deg.get(pid, 0))
            F[i, 1] = float(pr.get(pid, 0.0))
            F[i, 2] = float(andeg.get(pid, 0.0))
    return F, hit

def go_svd(go_gaf_gz: Path, union: list[str], alias_map: dict, k=128):
    import gzip
    gene2terms = {}
    with gzip.open(go_gaf_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            db_object_symbol = parts[2].upper()
            go_id = parts[4]
            g = alias_map.get(db_object_symbol, db_object_symbol)
            gene2terms.setdefault(g, set()).add(go_id)

    vocab = {}
    rows = []
    cols = []
    data = []
    for i, g in enumerate([u.upper() for u in union]):
        g2 = alias_map.get(g, g)
        terms = gene2terms.get(g2, set())
        for t in terms:
            j = vocab.setdefault(t, len(vocab))
            rows.append(i); cols.append(j); data.append(1.0)

    if len(vocab) == 0:
        return np.zeros((len(union), k), dtype=np.float32), 0

    from scipy.sparse import csr_matrix
    X = csr_matrix((data, (rows, cols)), shape=(len(union), len(vocab)), dtype=np.float32)

    svd = TruncatedSVD(n_components=min(k, max(2, min(X.shape)-1)), random_state=6)
    E = svd.fit_transform(X).astype(np.float32)
    # pad to k
    if E.shape[1] < k:
        E = np.pad(E, ((0,0),(0,k-E.shape[1])), mode="constant")
    hit = int((X.getnnz(axis=1) > 0).sum())
    return E, hit

def reactome_svd(uniprot2reactome_path: Path, union: list[str], alias_map: dict, uniprot_to_symbol: dict, k=128):
    df = pd.read_csv(uniprot2reactome_path, sep="\t", header=None,
                     names=["uniprot", "pathway", "url", "name", "evidence", "species"],
                     dtype=str)
    df = df[df["species"].astype(str).str.contains("Homo sapiens", na=False)]

    gene2paths = {}
    for u, p in zip(df["uniprot"], df["pathway"]):
        sym = uniprot_to_symbol.get(str(u), None)
        if sym:
            gene2paths.setdefault(sym.upper(), set()).add(p)

    vocab = {}
    rows=[]; cols=[]; data=[]
    for i, g in enumerate([u.upper() for u in union]):
        g2 = alias_map.get(g, g)
        paths = gene2paths.get(g2, set())
        for p in paths:
            j = vocab.setdefault(p, len(vocab))
            rows.append(i); cols.append(j); data.append(1.0)

    if len(vocab) == 0:
        return np.zeros((len(union), k), dtype=np.float32), 0

    from scipy.sparse import csr_matrix
    X = csr_matrix((data, (rows, cols)), shape=(len(union), len(vocab)), dtype=np.float32)

    svd = TruncatedSVD(n_components=min(k, max(2, min(X.shape)-1)), random_state=6)
    E = svd.fit_transform(X).astype(np.float32)
    if E.shape[1] < k:
        E = np.pad(E, ((0,0),(0,k-E.shape[1])), mode="constant")
    hit = int((X.getnnz(axis=1) > 0).sum())
    return E, hit

def main():
    union, gene_cols, train_genes = load_union_genes()

    hgnc_path = EXT / "hgnc" / "hgnc_complete_set.txt"
    alias_map, uniprot_to_symbol = build_hgnc_maps(hgnc_path)

    genept_dir = EXT / "genept" / "GenePT_emebdding_v2"
    m3_pkl  = genept_dir / "GenePT_gene_protein_embedding_model_3_text.pickle"
    ada_pkl = genept_dir / "GenePT_gene_embedding_ada_text.pickle"

    E_m3, hit_m3 = genept_pca(m3_pkl, union, alias_map, pca_dim=128)
    E_ada, hit_ada = genept_pca(ada_pkl, union, alias_map, pca_dim=128)

    np.save(OUT / "genept_m3_pca128.npy", E_m3)
    np.save(OUT / "genept_ada_pca128.npy", E_ada)

    aliases_txt = EXT / "string" / "9606.protein.aliases.v12.0.txt"
    links_txt   = EXT / "string" / "9606.protein.links.v12.0.txt"

    F_string, hit_string = string_graph_features(aliases_txt, links_txt, union, alias_map)
    np.save(OUT / "string_graph_feats.npy", F_string)

    go_gaf = EXT / "go" / "goa_human.gaf.gz"
    E_go, hit_go = go_svd(go_gaf, union, alias_map, k=128)
    np.save(OUT / "go_svd128.npy", E_go)

    reactome = EXT / "reactome" / "UniProt2Reactome.txt"
    E_re, hit_re = reactome_svd(reactome, union, alias_map, uniprot_to_symbol, k=128)
    np.save(OUT / "reactome_svd128.npy", E_re)

    # Save the exact union ordering used to build all .npy feature matrices
    np.save(OUT / "union_genes.npy", np.array(union, dtype=object))
    (OUT / "union_genes.txt").write_text("\n".join(union) + "\n")

    report = {
        "union_genes": len(union),
        "genept_m3_hits": int(hit_m3),
        "genept_ada_hits": int(hit_ada),
        "string_hits": int(hit_string),
        "go_hits": int(hit_go),
        "reactome_hits": int(hit_re),
        "paths": {k: str((OUT / k).resolve()) for k in [
            "genept_m3_pca128.npy",
            "genept_ada_pca128.npy",
            "string_graph_feats.npy",
            "go_svd128.npy",
            "reactome_svd128.npy",
        ]}
    }
    with open(OUT / "coverage_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
